# ComplianceAgent: validates roster_solution.csv against Fair Work rules
# Enforces full Fair Work Act rules on the solver’s relaxed roster,
# detecting violations and proposing swap-based fixes.

import os, json
import pandas as pd
from datetime import datetime, timedelta

class ComplianceAgent:
    def __init__(self,
                 roster_path="roster_solution.csv",
                 params_path="rostering_parameters.json",
                 out_dir=".",
                 fallback_year=2024):
        self.roster_path = roster_path
        self.params_path = params_path
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.fallback_year = fallback_year

        self.roster_df = None
        self.params = {}
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "inputs": {
                "roster_solution": roster_path,
                "rostering_parameters": params_path
            },
            "summary": {
                "assignments": 0,
                "employees": 0,
                "days": 0,
                "violations_count": 0,
                "violation_types": {}
            },
            "violations": [],
            "swaps": [],
            "notes": []
        }

  
    # Load artifacts
    # --------------
    def load(self):
        if not os.path.exists(self.roster_path):
            raise FileNotFoundError(f"Roster file not found: {self.roster_path}")
        if os.path.getsize(self.roster_path) == 0:
            raise ValueError("Roster file is empty. Ensure SolverAgent produced assignments.")

        self.roster_df = pd.read_csv(self.roster_path)
        if "hours" in self.roster_df.columns:
            self.roster_df["hours"] = pd.to_numeric(self.roster_df["hours"], errors="coerce").fillna(0.0)
        else:
            self.roster_df["hours"] = 0.0

        if os.path.exists(self.params_path):
            with open(self.params_path, "r", encoding="utf-8") as f:
                self.params = json.load(f)
        else:
            self.report["notes"].append("Parameters file not found. Using defaults.")

        self._normalize_dates()

        self.report["summary"]["assignments"] = int(len(self.roster_df))
        self.report["summary"]["employees"] = int(self.roster_df["employee"].nunique())
        self.report["summary"]["days"] = int(self.roster_df["date_parsed"].dt.date.nunique())

    def _normalize_dates(self):
        def parse_date_str(s):
            s = (s or "").strip()
            try:
                return datetime.fromisoformat(s)
            except Exception:
                pass
            try:
                base = datetime.strptime(s, "%a %b %d")
                return base.replace(year=self.fallback_year)
            except Exception:
                pass
            try:
                return datetime.strptime(s, "%Y-%m-%d")
            except Exception:
                return None

        def parse_time(s):
            try:
                hh, mm = s.split(":")
                return int(hh), int(mm)
            except Exception:
                return None

        parsed_dates, start_dt, end_dt = [], [], []
        for _, row in self.roster_df.iterrows():
            d = parse_date_str(row.get("date", ""))
            parsed_dates.append(d)
            st = parse_time(row.get("start_time", ""))
            et = parse_time(row.get("end_time", ""))
            if d and st and et:
                start_dt.append(d.replace(hour=st[0], minute=st[1]))
                end_dt.append(d.replace(hour=et[0], minute=et[1]))
            else:
                start_dt.append(None)
                end_dt.append(None)

        self.roster_df["date_parsed"] = parsed_dates
        self.roster_df["start_dt"] = start_dt
        self.roster_df["end_dt"] = end_dt

    
    # Compliance checks
    # -----------------
    def check_min_shift_length(self):
        min_hours = float(self._param_value("Minimum Hours Per Shift", default=3))
        for _, row in self.roster_df.iterrows():
            if row["hours"] < min_hours:
                self._add_violation("Shift Length", row, {"hours": row["hours"], "rule": f"Minimum {min_hours} hours"})

    def check_rest_periods(self):
        min_rest = float(self._param_value("Minimum Hours Between Shifts", default=10))
        for emp, group in self.roster_df.groupby("employee"):
            g = group.sort_values("date_parsed")
            prev_end = None
            for _, row in g.iterrows():
                if prev_end and row["start_dt"]:
                    rest_hours = (row["start_dt"] - prev_end).total_seconds() / 3600.0
                    if rest_hours < min_rest:
                        self._add_violation("Rest Period", row, {"rest_hours": round(rest_hours, 2),
                                                                "rule": f"Minimum {min_rest} hours"})
                if row["end_dt"]:
                    prev_end = row["end_dt"]

    def check_meal_breaks(self):
        for _, row in self.roster_df.iterrows():
            h = row["hours"]
            if h > 5:
                self._add_violation("Meal Break", row, {"hours": h, "rule": "30 min unpaid meal break required"})
            elif h > 4:
                self._add_violation("Rest Break", row, {"hours": h, "rule": "10 min paid rest break required"})

    def check_weekly_hours(self):
        bands = {"full-time": (35, 38), "part-time": (20, 32), "casual": (8, 24)}
        if "type" not in self.roster_df.columns: return
        df = self.roster_df.dropna(subset=["date_parsed"]).copy()
        df["iso_year"] = df["date_parsed"].dt.isocalendar().year
        df["iso_week"] = df["date_parsed"].dt.isocalendar().week
        for (emp, year, week), group in df.groupby(["employee", "iso_year", "iso_week"]):
            typ = str(group.iloc[0].get("type", "")).lower()
            if "full" in typ: band = bands["full-time"]
            elif "part" in typ: band = bands["part-time"]
            elif "casual" in typ: band = bands["casual"]
            else: continue
            total_hours = group["hours"].sum()
            if total_hours < band[0] or total_hours > band[1]:
                self.report["violations"].append({
                    "type": "Weekly Hours",
                    "employee": emp,
                    "iso_year": int(year),
                    "iso_week": int(week),
                    "total_hours": round(total_hours, 2),
                    "rule": f"{band[0]}–{band[1]} hours for {typ}"
                })

    
    # Swap suggestions
    # ----------------
    def suggest_swaps(self):
        for v in self.report["violations"]:
            day = v.get("date")
            emp = v.get("employee")
            row = self.roster_df[(self.roster_df["employee"] == emp) & (self.roster_df["date"] == day)]
            if row.empty: continue
            station = row.iloc[0].get("station", "")
            candidates = self.roster_df[(self.roster_df["date"] == day) & (self.roster_df["station"] == station)]
            candidates = candidates[candidates["employee"] != emp]
            if not candidates.empty:
                cand = candidates.iloc[0]
                self.report["swaps"].append({
                    "violation": v,
                    "suggested_employee": cand["employee"],
                    "station": station,
                    "date": day
                })

    
    # Utilities
    # ---------
    def _param_value(self, name, default=None):
        cell = self.params.get("basic", {}).get(name, {})
        try:
            return float(cell.get("value", default))
        except Exception:
            return default

    def _add_violation(self, vtype, row, extra=None):
        item = {
            "type": vtype,
            "employee": row.get("employee"),
            "date": row.get("date"),
            "station": row.get("station"),
            "code": row.get("code"),
        }
        if extra: item.update(extra)
        self.report["violations"].append(item)

    
    # Save report
    # -----------
    def save(self):
        self.report["summary"]["violations_count"] = len(self.report["violations"])
        vt = {}
        for v in self.report["violations"]:
            vt[v["type"]] = vt.get(v["type"], 0) + 1
        self.report["summary"]["violation_types"] = vt
        out_path = os.path.join(self.out_dir, "compliance_report.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        print("Compliance report written ->", out_path)

   
    # End-to-end run
    # --------------
    def run(self):
        print("Running ComplianceAgent...")
        self.load()
        if self.roster_df.empty:
            self.report["notes"].append("Roster has 0 assignments; skipping checks.")
            self.save()
            return self.report

        # Perform compliance checks in order
        self.check_min_shift_length()
        self.check_rest_periods()
        self.check_meal_breaks()
        self.check_weekly_hours()
        self.suggest_swaps()

        # Save report to JSON
        self.save()
        return self.report