# Preprocessor Agent
# Wraps preprocessing into a class that uses ingested context and writes cleaned artifacts + manifest.

import os, re, json, pickle
from datetime import datetime
import pandas as pd

class PreprocessorAgent:
    def __init__(self, ingested_path="ingested.pkl", out_dir="."):
        self.ingested_path = ingested_path
        self.out_dir = out_dir
        self.ingested = self._safe_load_ingested()
        self.manifest = {
            "timestamp": self._now_iso(),
            "inputs": ingested_path,
            "outputs": {},
            "warnings": []
        }
        os.makedirs(out_dir, exist_ok=True)

        # Config
        self.LEGEND_TOKENS = {
            "Employee Types:", "Shift Colors:",
            "Gray = Day Off", "Green = Full-Time", "Light Green = 1F",
            "Light Orange = 3F", "Orange = Casual", "Yellow = 2F", "Yellow = Part-Time"
        }
        self.MEETING_TIME = ("09:00", "17:00")  # maps code 'M' to explicit times

    
    # Helpers
    # -------
    def _safe_load_ingested(self):
        if os.path.exists(self.ingested_path):
            with open(self.ingested_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _write_csv(self, df, path):
        df.to_csv(path, index=False)
        print(f"Saved {path} (rows={len(df)})")

    def _now_iso(self):
        return datetime.utcnow().isoformat() + "Z"

    def _safe_str(self, x):
        return "" if x is None else str(x)

    
    # Availability cleaning (with audit)
    # ----------------------------------
    def clean_availability(self, out_path="availability_cleaned_for_norm.csv"):
        warnings = []
        avail_tbl, src = None, None

        # find availability-like sheet
        for fname, sheets in self.ingested.items():
            if not isinstance(sheets, dict): continue
            for sheet_name, obj in sheets.items():
                tbl = obj.get("table")
                if isinstance(tbl, pd.DataFrame) and not tbl.empty:
                    cols = [str(c).lower() for c in tbl.columns]
                    if any("employee" in c for c in cols) and any(tok in " ".join(cols) for tok in ("mon","tue","wed","dec","date")):
                        avail_tbl = tbl.copy()
                        src = f"{fname} / {sheet_name}"
                        break
            if avail_tbl is not None:
                break
        if avail_tbl is None:
            raise RuntimeError("Availability table not found")

        meta_cols = list(avail_tbl.columns[:4])
        date_cols = [c for c in avail_tbl.columns if c not in meta_cols]
        clean = avail_tbl.copy()

        # remove explicit legend tokens and heuristic legend-like strings
        legend_pattern = re.compile(r":|=|employee types|shift colors|legend|gray =|yellow =|light", flags=re.IGNORECASE)
        removed_count = 0
        for c in date_cols:
            clean[c] = clean[c].astype(str).str.strip()
            mask_bad = clean[c].isin(self.LEGEND_TOKENS)
            removed_count += int(mask_bad.sum())
            clean.loc[mask_bad, c] = ""
            mask_legend = clean[c].str.contains(legend_pattern, na=False)
            removed_count += int(mask_legend.sum())
            clean.loc[mask_legend, c] = ""
            # keep only tokens shaped like alphanumerics or '/' (remove anything else)
            mask_invalid = ~clean[c].str.match(r"^[A-Za-z0-9/]+$", na=False)
            # don't blank legitimate blanks
            mask_invalid &= (clean[c] != "")
            invalid_count = int(mask_invalid.sum())
            if invalid_count:
                warnings.append(f"Column '{c}': removed {invalid_count} non-token cells")
            clean.loc[mask_invalid, c] = ""

        if removed_count:
            warnings.append(f"Removed {removed_count} legend/notes cells from availability")

        # audit tokens
        tokens = set()
        for c in date_cols:
            tokens.update(clean[c].dropna().astype(str).str.strip().unique().tolist())
        tokens = sorted([t for t in tokens if t not in ("", "nan")])
        audit_path = out_path.replace(".csv", "_tokens_audit.txt")
        with open(audit_path, "w", encoding="utf-8") as f:
            f.write("meta_cols: " + ", ".join(meta_cols) + "\n")
            f.write("date_cols_count: " + str(len(date_cols)) + "\n")
            f.write("unique_tokens:\n")
            for t in tokens:
                f.write(f"- {t}\n")
        warnings.append(f"Availability tokens audit written to {audit_path}")

        self._write_csv(clean, out_path)
        self.manifest["outputs"]["availability"] = out_path
        self.manifest["outputs"]["availability_tokens_audit"] = audit_path
        self.manifest["warnings"].extend(warnings)

    
    # Shift codes cleaning (canonicalize, filter, map)
    # ------------------------------------------------
    def clean_shift_codes(self, out_path="shift_codes_cleaned.csv"):
        warnings = []
        shift_tbl, source = None, None

        # find shift codes sheet
        for fname, sheets in self.ingested.items():
            if not isinstance(sheets, dict): continue
            for sheet_name, obj in sheets.items():
                tbl = obj.get("table")
                if isinstance(tbl, pd.DataFrame) and not tbl.empty:
                    cols = [str(c).lower() for c in tbl.columns]
                    if any("code" in c for c in cols) and any("time" in c for c in cols):
                        shift_tbl = tbl.copy()
                        source = f"{fname} / {sheet_name}"
                        break
            if shift_tbl is not None:
                break
        if shift_tbl is None:
            raise RuntimeError("Shift codes table not found")

        # detect primary columns
        code_col = [c for c in shift_tbl.columns if "code" in str(c).lower()][0]
        time_col = [c for c in shift_tbl.columns if "time" in str(c).lower()][0]

        # normalize and filter to code-like rows
        shift_tbl[code_col] = shift_tbl[code_col].astype(str).str.strip().replace({"nan": ""})
        mask_code_like = shift_tbl[code_col].astype(str).str.match(r"^[A-Za-z0-9/]+$", na=False)
        dropped = int((~mask_code_like).sum())
        if dropped:
            warnings.append(f"Dropped {dropped} non-code rows from shift codes (descriptive text removed)")
        valid = shift_tbl.loc[mask_code_like].copy()

        # parse time field
        def _split_time(t):
            if not isinstance(t, str): return (None, None, "empty")
            t = t.strip()
            if t in ("-", "\\-", ""): return (None, None, "dash")
            if t.lower().startswith("varies"): return (None, None, "varies")
            if "-" in t:
                a, b = t.split("-", 1); return a.strip(), b.strip(), "range"
            if "–" in t:
                a, b = t.split("–", 1); return a.strip(), b.strip(), "range"
            if " to " in t.lower():
                a, b = re.split(r"\s+to\s+", t, flags=re.IGNORECASE); return a.strip(), b.strip(), "range"
            return (t if t else None, None, "single")

        parsed = valid[time_col].astype(str).map(_split_time)
        starts, ends, kinds = zip(*parsed)
        valid = valid.assign(start_time=list(starts), end_time=list(ends), time_kind=list(kinds))

        # canonicalize column names for clarity
        cols = list(valid.columns)
        if len(cols) >= 2:
            valid = valid.rename(columns={cols[0]: "code", cols[1]: "time"})

        # trim code column
        if "code" in valid.columns:
            valid["code"] = valid["code"].astype(str).str.strip().replace({"nan": ""})

        # mapping rules
        if "code" in valid.columns:
            if self.MEETING_TIME and self.MEETING_TIME[0] and self.MEETING_TIME[1]:
                valid.loc[valid["code"] == "M", ["start_time", "end_time", "time_kind"]] = [self.MEETING_TIME[0], self.MEETING_TIME[1], "range"]
            valid.loc[valid["code"] == "/", ["start_time", "end_time", "time_kind"]] = [None, None, "dash"]

        # drop empty-code defensively
        before = len(valid)
        valid = valid[valid["code"].astype(str).str.strip() != ""].copy()
        after = len(valid)
        if before != after:
            warnings.append(f"Dropped {before-after} empty-code rows after parsing")

        # reorder columns
        cols_out = []
        for c in ["code", "time", "time_kind", "start_time", "end_time"]:
            if c in valid.columns: cols_out.append(c)
        cols_out += [c for c in valid.columns if c not in cols_out]
        valid = valid[cols_out]

        self._write_csv(valid, out_path)
        non_range = valid[valid["time_kind"] != "range"]
        if not non_range.empty:
            warnings.append(f"{len(non_range)} shift codes have non-range times (e.g., 'Varies' or '-')")

        self.manifest["outputs"]["shift_codes"] = out_path
        self.manifest["warnings"].extend(warnings)

    
    # Fixed hours tidy (multi-row header collapse)
    # --------------------------------------------
    def tidy_fixed_hours(self, out_path="fixed_hours_tidy.csv"):
        warnings = []
        obj, df, src = None, None, None

        # locate expected sheet or any matching "fixed hours"
        expected_file = "fixed_hours_template_2columns.xlsx"
        expected_sheet = "Fixed Hours Template - Table 1"
        if expected_file in self.ingested and expected_sheet in self.ingested[expected_file]:
            obj = self.ingested[expected_file][expected_sheet]
            df = obj.get("table").copy()
            src = f"{expected_file} / {expected_sheet}"
        else:
            for fname, sheets in self.ingested.items():
                for sname, o in sheets.items():
                    if "fixed hours" in str(sname).lower() or "fixed working hours" in str(sname).lower():
                        obj = o
                        df = o.get("table").copy()
                        src = f"{fname} / {sname}"
                        break
                if obj is not None: break
        if obj is None or df is None or df.empty:
            raise RuntimeError("Fixed hours template sheet not found in ingested object")

        # attempt header collapse using raw_preview and original file if available
        data_raw = df.copy()
        rp = obj.get("raw_preview")
        try:
            if isinstance(rp, pd.DataFrame) and not rp.empty and os.path.exists(expected_file):
                full_raw = pd.read_excel(expected_file, sheet_name=expected_sheet, header=None, dtype=str).fillna("")
                # heuristic to find header rows
                hdr_idx = None
                for i in range(max(1, len(full_raw)-1)):
                    row0 = " ".join(full_raw.iloc[i].astype(str).str.lower().tolist())
                    row1 = " ".join(full_raw.iloc[i+1].astype(str).str.lower().tolist())
                    if "fixed hours type" in row0 or "weekly schedule" in row0 or "std" in row1 or "rest" in row1:
                        hdr_idx = i+1
                        break
                if hdr_idx is not None:
                    r1 = full_raw.iloc[hdr_idx-1].astype(str).tolist()
                    r2 = full_raw.iloc[hdr_idx].astype(str).tolist()
                    combined = []
                    for a, b in zip(r1, r2):
                        a_clean, b_clean = a.strip(), b.strip()
                        if a_clean and b_clean and a_clean.lower() != "nan":
                            combined.append(f"{a_clean} {b_clean}".strip())
                        elif b_clean and b_clean.lower() != "nan":
                            combined.append(b_clean)
                        elif a_clean and a_clean.lower() != "nan":
                            combined.append(a_clean)
                        else:
                            combined.append("")
                    headers = [re.sub(r"\s+", " ", h).strip() or f"col_{i}" for i, h in enumerate(combined)]
                    data_start = hdr_idx + 1
                    data_raw = full_raw.iloc[data_start:].reset_index(drop=True)
                    data_raw.columns = headers
                    data_raw = data_raw.dropna(how="all").reset_index(drop=True)
        except Exception as e:
            warnings.append(f"Header collapse fallback used: {e}")

        # clean column names (dedupe, strip)
        cols = list(data_raw.columns)
        clean_cols, seen = [], {}
        for i, c in enumerate(cols):
            s = re.sub(r"\bnan(_\d+)?\b", "", str(c), flags=re.IGNORECASE).strip()
            s = re.sub(r"\s+", " ", s)
            if s == "": s = f"col_{i}"
            idx = seen.get(s, 0)
            name = s if idx == 0 else f"{s}_{idx}"
            seen[s] = idx + 1
            clean_cols.append(name)
        data_raw.columns = clean_cols

        # detect template column
        template_col = None
        for c in data_raw.columns:
            if "fixed hours type" in c.lower() or c.lower() == "type" or "template" in c.lower():
                template_col = c
                break
        if template_col is None:
            template_col = data_raw.columns[0]

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_cols = [c for c in data_raw.columns if any(d.lower() in c.lower() for d in day_names)]

        tidy_rows = []
        for _, r in data_raw.iterrows():
            template_name = self._safe_str(r[template_col]).strip()
            if template_name == "" or template_name.lower() in ("template information", "weekly schedule"):
                continue
            for d in day_names:
                matches = [c for c in day_cols if d.lower() in c.lower()]
                if not matches:
                    continue
                std_val, rest_val = None, None
                for m in matches:
                    ml = m.lower()
                    if "std" in ml:
                        std_val = r[m]
                    elif "rest" in ml:
                        rest_val = r[m]
                    else:
                        if std_val is None:
                            std_val = r[m]
                        elif rest_val is None:
                            rest_val = r[m]
                tidy_rows.append({
                    "template_name": template_name,
                    "day": d,
                    "std_hours": pd.to_numeric(std_val, errors="coerce"),
                    "rest_hours": pd.to_numeric(rest_val, errors="coerce")
                })

        fixed_hours_tidy = pd.DataFrame(tidy_rows)
        before = len(fixed_hours_tidy)
        fixed_hours_tidy = fixed_hours_tidy.dropna(subset=["std_hours", "rest_hours"], how="all").reset_index(drop=True)
        after = len(fixed_hours_tidy)
        if before != after:
            warnings.append(f"Dropped {before-after} empty template rows from fixed hours tidy")

        self._write_csv(fixed_hours_tidy, out_path)
        self.manifest["outputs"]["fixed_hours"] = out_path
        self.manifest["warnings"].extend(warnings)

    
    # Run all and write manifest
    # -------------------------
    def run(self):
        print("Running PreprocessorAgent...")
        try:
            self.clean_availability()
        except Exception as e:
            self.manifest["outputs"]["availability_error"] = str(e)
            self.manifest["warnings"].append(f"Availability error: {e}")

        try:
            self.clean_shift_codes()
        except Exception as e:
            self.manifest["outputs"]["shift_codes_error"] = str(e)
            self.manifest["warnings"].append(f"Shift codes error: {e}")

        try:
            self.tidy_fixed_hours()
        except Exception as e:
            self.manifest["outputs"]["fixed_hours_error"] = str(e)
            self.manifest["warnings"].append(f"Fixed hours error: {e}")

        manifest_path = os.path.join(self.out_dir, "preprocessor_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)
        print("Wrote manifest:", manifest_path)
        print("Preprocessor finished. Manifest summary:")
        print(json.dumps(self.manifest, indent=2))
        return self.manifest


