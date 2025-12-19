# Normalization Agent
# Converts cleaned availability and shift‑code data into a unified, time‑expanded tidy format ready for constraint solving

import pandas as pd, json, os
from datetime import datetime

class NormalizationAgent:
    def __init__(self,
                 availability_path="availability_cleaned_for_norm.csv",
                 shift_codes_path="shift_codes_cleaned.csv",
                 out_dir="."):
        self.availability_path = availability_path
        self.shift_codes_path = shift_codes_path
        self.out_dir = out_dir
        self.manifest = {
            "timestamp": datetime.now().isoformat(),
            "inputs": {
                "availability": availability_path,
                "shift_codes": shift_codes_path
            },
            "outputs": {},
            "warnings": [],
            "unmapped_tokens": []
        }
        os.makedirs(out_dir, exist_ok=True)

    def _load_csv(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        return pd.read_csv(path, dtype=str).fillna("")

    def _parse_shift_codes(self):
        df = self._load_csv(self.shift_codes_path)
        df["code"] = df["code"].astype(str).str.strip()
        df["start_time"] = df["start_time"].replace("", pd.NA)
        df["end_time"] = df["end_time"].replace("", pd.NA)
        df["time_kind"] = df["time_kind"].fillna("unknown")
        df = df.dropna(subset=["code"])
        return df.set_index("code")

    def _compute_hours(self, start, end):
        try:
            h1 = pd.to_datetime(start, format="%H:%M")
            h2 = pd.to_datetime(end, format="%H:%M")
            return round((h2 - h1).total_seconds() / 3600, 2)
        except Exception:
            return None

    def normalize_availability(self):
        avail = self._load_csv(self.availability_path)
        shift_map = self._parse_shift_codes()
        meta_cols = list(avail.columns[:4])
        date_cols = [c for c in avail.columns if c not in meta_cols]

        tidy_rows = []
        unmapped = set()

        for _, row in avail.iterrows():
            meta = {k: row[k] for k in meta_cols}
            for date in date_cols:
                token = str(row[date]).strip()
                if token == "" or token.lower() in ("nan", "n/a"):
                    continue
                entry = {**meta, "date": date, "token": token}
                if token in shift_map.index:
                    shift = shift_map.loc[token]
                    entry["start_time"] = shift.get("start_time", "")
                    entry["end_time"] = shift.get("end_time", "")
                    entry["time_kind"] = shift.get("time_kind", "")
                    entry["hours"] = self._compute_hours(shift.get("start_time"), shift.get("end_time"))
                else:
                    entry["start_time"] = ""
                    entry["end_time"] = ""
                    entry["time_kind"] = "unmapped"
                    entry["hours"] = None
                    unmapped.add(token)
                tidy_rows.append(entry)

        tidy_df = pd.DataFrame(tidy_rows)
        tidy_path = os.path.join(self.out_dir, "availability_tidy.csv")
        tidy_df.to_csv(tidy_path, index=False)
        self.manifest["outputs"]["availability_tidy"] = tidy_path
        self.manifest["unmapped_tokens"] = sorted(unmapped)
        if unmapped:
            self.manifest["warnings"].append(f"{len(unmapped)} tokens were not mapped to shift codes")

    def run(self):
        print("Running NormalizationAgent...")
        try:
            self.normalize_availability()
        except Exception as e:
            self.manifest["warnings"].append(f"Normalization error: {e}")
        report_path = os.path.join(self.out_dir, "normalization_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)
        print("Wrote normalization report:", report_path)
        print("Normalization finished. Report summary:")
        print(json.dumps(self.manifest, indent=2))
        return self.manifest


