# SolverAgent: CP-SAT roster optimization
# Generates a feasible roster using relaxed CP‑SAT constraints—ensuring assignments exist—
# while deferring full Fair Work compliance enforcement to the downstream Compliance Agent

import os, json
import pandas as pd
from datetime import datetime
from collections import defaultdict
from ortools.sat.python import cp_model

class SolverAgent:
    def __init__(self,
                 availability_path="availability_tidy.csv",
                 shift_codes_path="shift_codes_cleaned.csv",
                 fixed_hours_path="fixed_hours_tidy.csv",
                 params_path="rostering_parameters.json",
                 out_dir=".",
                 time_limit_sec=120,
                 num_workers=8):
        self.availability_path = availability_path
        self.shift_codes_path = shift_codes_path
        self.fixed_hours_path = fixed_hours_path
        self.params_path = params_path
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.time_limit_sec = time_limit_sec
        self.num_workers = num_workers

        self.model = cp_model.CpModel()
        self.assign = {}
        self.hours = {}
        self.start_min = {}
        self.end_min = {}
        self.emp_days = defaultdict(set)
        self.coverage_vars = {}
        self.manifest = {
            "timestamp": datetime.now().isoformat(),
            "inputs": {
                "availability_tidy": availability_path,
                "shift_codes_cleaned": shift_codes_path,
                "fixed_hours_tidy": fixed_hours_path,
                "rostering_parameters": params_path
            },
            "warnings": [],
            "outputs": {},
            "objective_weights": {
                "coverage": 5.0,
                "fairness": 0.5
            }
        }


        # RELAXED defaults for feasibility
        # --------------------------------
        self.MIN_SHIFT_HOURS = 3.0
        self.MIN_REST_HOURS = 8.0       # RELAXED: was 10h, now 8h
        self.MAX_CONSEC_DAYS = 6
        self.MIN_STAFF_COUNT = 1        # RELAXED: was 2, now 1
        self.MIN_FULLTIME_ON_SHIFT = 0  # RELAXED: was 1, now 0
        self.service_periods = []


    # Helpers
    # -------
    def _load_csv(self, path, required=False):
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Missing required file: {path}")
            self.manifest["warnings"].append(f"Optional file not found: {path}")
            return None
        return pd.read_csv(path, dtype=str).fillna("")

    def _load_json(self, path, required=False):
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Missing required JSON: {path}")
            self.manifest["warnings"].append(f"Optional JSON not found: {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _parse_hhmm_to_min(self, s):
        try:
            hh, mm = s.strip().split(":")
            return int(hh) * 60 + int(mm)
        except Exception:
            return None

    def _hours_between(self, start_str, end_str):
        s = self._parse_hhmm_to_min(start_str)
        e = self._parse_hhmm_to_min(end_str)
        if s is None or e is None or e < s:
            return None
        return round((e - s) / 60.0, 2)

    def _date_str(self, label):
        return str(label).strip()


    # Load artifacts and parameters
    # -----------------------------
    def load(self):
        self.avail_df = self._load_csv(self.availability_path, required=True)
        self.shift_df = self._load_csv(self.shift_codes_path, required=True)
        self.fixed_df = self._load_csv(self.fixed_hours_path, required=False)
        self.params = self._load_json(self.params_path, required=True)

        for df in [self.avail_df, self.shift_df]:
            df.columns = [c.strip() for c in df.columns]

        # Extract service periods
        self.service_periods = []
        for r in self.params.get("service_periods", []):
            if r.get("period") and r.get("start") and r.get("end"):
                self.service_periods.append({"name": r["period"], "start": r["start"], "end": r["end"]})

        # Identify key columns
        emp_cols = [c for c in self.avail_df.columns if c.lower() in ["employee", "employee name", "name"]]
        type_cols = [c for c in self.avail_df.columns if "type" in c.lower()]
        station_cols = [c for c in self.avail_df.columns if "station" in c.lower() or "position" in c.lower()]
        self.emp_col = emp_cols[0] if emp_cols else None
        self.type_col = type_cols[0] if type_cols else None
        self.station_col = station_cols[0] if station_cols else None


    # Build decision variables
    # ------------------------
    def build_vars(self):
        for idx, r in self.avail_df.iterrows():
            emp = r.get(self.emp_col, f"emp_{idx}") if self.emp_col else f"emp_{idx}"
            day = self._date_str(r.get("date"))
            token = str(r.get("token")).strip()
            if token == "":
                continue

            start_t = str(r.get("start_time","")).strip()
            end_t = str(r.get("end_time","")).strip()
            h = r.get("hours", "")
            try:
                hours = float(h)
            except Exception:
                hours = self._hours_between(start_t, end_t) or 0.0

            smin = self._parse_hhmm_to_min(start_t)
            emin = self._parse_hhmm_to_min(end_t)
            if smin is None or emin is None or emin < smin:
                continue

            var = self.model.NewBoolVar(f"x_{emp}_{day}_{token}")
            key = (emp, day, token)
            self.assign[key] = var
            self.hours[key] = hours
            self.start_min[key] = smin
            self.end_min[key] = emin
            self.emp_days[emp].add(day)


    # Hard constraints (relaxed)
    # --------------------------
    def add_hard_constraints(self):
        # 1) One shift per employee per day
        by_emp_day = defaultdict(list)
        for (emp, day, token), v in self.assign.items():
            by_emp_day[(emp, day)].append(v)
        for (emp, day), vars_list in by_emp_day.items():
            self.model.Add(sum(vars_list) <= 1)

        # 2) Rest period simplified
        # RELAXED: only forbid direct overlaps, not full 10h enforcement
        for emp, days in self.emp_days.items():
            days_sorted = sorted(list(days))
            for i in range(len(days_sorted)-1):
                d1, d2 = days_sorted[i], days_sorted[i+1]
                d1_vars = [self.assign[(emp, d1, t)] for (e, dd, t) in self.assign.keys() if e==emp and dd==d1]
                d2_vars = [self.assign[(emp, d2, t)] for (e, dd, t) in self.assign.keys() if e==emp and dd==d2]
                for v1 in d1_vars:
                    for v2 in d2_vars:
                        self.model.Add(v1 + v2 <= 1)  # RELAXED

        # 3) Coverage (RELAXED: warn if no contributors, don’t block roster)
        def overlaps(s1, e1, s2, e2):
            return not (e2 <= s1 or e1 <= s2)

        all_days = sorted(set([d for (_, d, _) in self.assign.keys()]))
        for d in all_days:
            for p in self.service_periods:
                p_start = self._parse_hhmm_to_min(p["start"])
                p_end = self._parse_hhmm_to_min(p["end"])
                cov = self.model.NewIntVar(0, 1000, f"cov_{p['name']}_{d}")
                contributors = []
                for (emp, day, token), v in self.assign.items():
                    if day != d: continue
                    smin = self.start_min[(emp, day, token)]
                    emin = self.end_min[(emp, day, token)]
                    if overlaps(p_start, p_end, smin, emin):
                        contributors.append(v)
                if contributors:
                    self.model.Add(cov == sum(contributors))
                else:
                    self.model.Add(cov == 0)
                    self.manifest["warnings"].append(f"No coverage for {p['name']} on {d}")
                self.coverage_vars[(p["name"], d)] = cov


    # Objective (coverage + fairness + base reward)
    # ---------------------------------------------
    def add_objective(self):
        terms = []

        # Coverage reward
        for ((pname, d), cov) in self.coverage_vars.items():
            terms.append(self.manifest["objective_weights"]["coverage"] * cov)

        # Fairness reward (spread assignments across employees)
        fairness_w = self.manifest["objective_weights"]["fairness"]
        by_emp = defaultdict(list)
        for (emp, day, token), v in self.assign.items():
            by_emp[emp].append(v)
        for emp, vs in by_emp.items():
            terms.append(int(round(fairness_w)) * sum(vs))

        # Base reward per assignment (RELAXED: ensures non-empty roster)
        for (emp, day, token), v in self.assign.items():
            terms.append(2 * v)  # each assignment gives +2 reward

        # Maximize coverage + fairness + base reward
        self.model.Maximize(sum(terms))



    # Solve and export
    # ----------------
    def solve(self):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(self.time_limit_sec)
        solver.parameters.num_search_workers = int(self.num_workers)

        status = solver.Solve(self.model)
        self.manifest["solver_status"] = solver.StatusName(status)

        selected = [(k,v) for k,v in self.assign.items() if solver.BooleanValue(v)]
        print("Total assignment variables:", len(self.assign))
        print("Selected assignments:", len(selected))

        if not selected:
            print("⚠️ No assignments selected. Constraints may still be too strict.")
            print("Check: MIN_SHIFT_HOURS =", self.MIN_SHIFT_HOURS)
            print("Check: MIN_REST_HOURS =", self.MIN_REST_HOURS)
            print("Check: MIN_STAFF_COUNT =", self.MIN_STAFF_COUNT)
            print("Check: MIN_FULLTIME_ON_SHIFT =", self.MIN_FULLTIME_ON_SHIFT)

        # Extract solution rows
        sol_rows = []
        for (emp, day, token), v in self.assign.items():
            if solver.BooleanValue(v):
                row = self.avail_df[(self.avail_df["date"].astype(str)==day) &
                                    (self.avail_df["token"].astype(str)==token)]
                typ = row.iloc[0].get(self.type_col,"") if not row.empty and self.type_col else ""
                station = row.iloc[0].get(self.station_col,"") if not row.empty and self.station_col else ""
                start_t = row.iloc[0].get("start_time","") if not row.empty else ""
                end_t = row.iloc[0].get("end_time","") if not row.empty else ""
                sol_rows.append({
                    "employee": emp,
                    "type": typ,
                    "station": station,
                    "date": day,
                    "code": token,
                    "start_time": start_t,
                    "end_time": end_t,
                    "hours": round(self.hours[(emp, day, token)], 2)
                })

        out_path = os.path.join(self.out_dir, "roster_solution.csv")
        pd.DataFrame(sol_rows).to_csv(out_path, index=False)
        self.manifest["outputs"]["roster_solution"] = out_path

        # Coverage summary
        cov_summary = []
        for ((pname, d), cov) in self.coverage_vars.items():
            cov_summary.append({"period": pname, "day": d, "coverage": int(solver.Value(cov))})
        self.manifest["coverage_summary"] = cov_summary

        # Write manifest
        man_path = os.path.join(self.out_dir, "solver_manifest.json")
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

        print("Wrote roster_solution.csv ->", out_path)
        print("Wrote solver_manifest.json ->", man_path)
        print("Status:", self.manifest["solver_status"])
        return status


    # End-to-end run
    # --------------
    def run(self):
        print("Running SolverAgent (CP-SAT)...")
        self.load()
        self.build_vars()
        self.add_hard_constraints()
        self.add_objective()
        status = self.solve()
        return status



