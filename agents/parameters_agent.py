# Parameters Agent
# Extracts all scheduling rules and compliance parameters into a unified JSON configuration.

import os, json, pickle
import pandas as pd
from datetime import datetime

class ParametersAgent:
    def __init__(self, ingested_path="ingested.pkl", out_dir="."):
        self.ingested_path = ingested_path
        self.out_dir = out_dir
        self.ingested = self._safe_load_ingested()
        self.manifest = {
            "timestamp": datetime.now().isoformat(),
            "inputs": ingested_path,
            "outputs": {},
            "warnings": []
        }
        os.makedirs(out_dir, exist_ok=True)

    def _safe_load_ingested(self):
        if os.path.exists(self.ingested_path):
            with open(self.ingested_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _write_json(self, data, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {path}")

    def extract_parameters(self):
        parameters = {
            "basic": {},
            "service_periods": [],
            "compliance_notes": [],
            "other_parameters": {}
        }

        for fname, sheets in self.ingested.items():
            if not isinstance(sheets, dict): continue
            for sname, obj in sheets.items():
                tbl = obj.get("table")
                if not isinstance(tbl, pd.DataFrame) or tbl.empty:
                    continue
                sname_lower = sname.lower()

                # Basic Parameters
                if "basic parameters" in sname_lower:
                    for _, row in tbl.iterrows():
                        pname = str(row.get("Parameter Name","")).strip()
                        val = str(row.get("Value","")).strip()
                        unit = str(row.get("Unit","")).strip()
                        note = str(row.get("Notes","")).strip()
                        if pname:
                            parameters["basic"][pname] = {"value": val, "unit": unit, "notes": note}

                # Service Periods
                elif "service periods" in sname_lower:
                    for _, row in tbl.iterrows():
                        period = str(row.get("Service Period","")).strip()
                        start = str(row.get("Start Time","")).strip()
                        end = str(row.get("End Time","")).strip()
                        desc = str(row.get("Description","")).strip()
                        if period:
                            parameters["service_periods"].append({
                                "period": period, "start": start, "end": end, "description": desc
                            })

                # Compliance Notes
                elif "compliance notes" in sname_lower:
                    for _, row in tbl.iterrows():
                        note = " ".join([str(x) for x in row.tolist() if str(x).strip() != ""])
                        if note:
                            parameters["compliance_notes"].append(note)

                # Inject Melbourne Cup Day if mentioned
                elif any("Melbourne Cup Day" in n for n in parameters["compliance_notes"]):
                    parameters.setdefault("public_holidays", []).append({
                        "name": "Melbourne Cup Day",
                        "date": "2025-11-04",
                        "penalty_multiplier": 2.25
                    })

                # Other parameter-like sheets
                elif any(k in sname_lower for k in ["parameter","config","rule","note","summary"]):
                    parameters["other_parameters"][f"{fname}/{sname}"] = tbl.to_dict(orient="records")

        return parameters

    def run(self):
        print("Running ParametersAgent...")
        try:
            params = self.extract_parameters()
            out_path = os.path.join(self.out_dir, "rostering_parameters.json")
            self._write_json(params, out_path)
            self.manifest["outputs"]["rostering_parameters"] = out_path
        except Exception as e:
            self.manifest["warnings"].append(f"Parameters extraction error: {e}")

        manifest_path = os.path.join(self.out_dir, "parameters_manifest.json")
        self._write_json(self.manifest, manifest_path)
        print("ParametersAgent finished. Manifest summary:")
        print(json.dumps(self.manifest, indent=2))
        return self.manifest

