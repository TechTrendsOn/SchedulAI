# Ingestion Agent
# Reads all Excel and CSV inputs, cleans messy headers, extracts tables and notes,
# and produces a unified ingested dataset for the entire pipeline.

import os, re, json, pickle
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd


# Helper regex
# ------------
HTML_TOKEN_RE = re.compile(r"<[^>]+>")
COLAB_TOKEN_RE = re.compile(r"\.colab-[\w\-]+|colab-df-[\w\-]+")

def strip_html_tokens(x):
    if x is None:
        return ""
    s = str(x)
    s = HTML_TOKEN_RE.sub(" ", s)
    s = COLAB_TOKEN_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def sanitize_headers(headers: List[Any]) -> List[str]:
    out = []
    for h in headers:
        s = strip_html_tokens(h).replace("\n", " ").strip()
        s = re.sub(r"\s+", " ", s)
        out.append(s if s != "" else "unnamed")
    seen = {}
    res = []
    for c in out:
        idx = seen.get(c, 0)
        name = c if idx == 0 else f"{c}_{idx}"
        seen[c] = idx + 1
        res.append(name)
    return res

def clean_table(df: pd.DataFrame, min_cols: int = 1) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    df = df.dropna(how="all")
    if df.empty:
        return df.reset_index(drop=True)
    df = df[df.count(axis=1) >= min_cols].reset_index(drop=True)
    return df

def detect_header_row(raw: pd.DataFrame, expected_tokens=None, min_matches=2):
    if expected_tokens is None:
        expected_tokens = ["id","employee","name","type","station","shift","mon","tue","wed","date","code","time"]
    raw = raw.fillna("").astype(str)
    best_idx, best_score = None, -1.0
    for i in range(min(len(raw), 20)):
        row = [strip_html_tokens(x).strip().lower() for x in raw.iloc[i].tolist()]
        score = sum(1 for t in expected_tokens if any(t in c for c in row if c))
        score += 0.08 * sum(1 for c in row if c)
        if score > best_score:
            best_score, best_idx = score, i
    return int(best_idx) if best_score >= min_matches else None

def collapse_multirow_header(raw: pd.DataFrame, max_rows=4):
    n = min(max_rows, len(raw))
    nonempty_counts = [raw.iloc[r].notna().sum() for r in range(n)]
    for r in range(n-1):
        if nonempty_counts[r] >= 2 and nonempty_counts[r+1] >= 2:
            hdr1 = raw.iloc[r].fillna("").astype(str).tolist()
            hdr2 = raw.iloc[r+1].fillna("").astype(str).tolist()
            combined = [f"{strip_html_tokens(a)} {strip_html_tokens(b)}".strip() for a,b in zip(hdr1, hdr2)]
            return r+1, combined
    return None, None

def extract_notes_below_table(raw: pd.DataFrame) -> List[str]:
    raw = raw.fillna("").astype(str)
    nonempty = raw.apply(lambda r: sum(1 for v in r if str(v).strip() != ""), axis=1)
    if nonempty.empty:
        return []
    last_table_idx = nonempty[nonempty >= 2].index.max() if (nonempty >= 2).any() else -1
    notes_rows = raw.iloc[last_table_idx+1:] if (last_table_idx+1) < len(raw) else pd.DataFrame()
    notes = []
    for _, r in notes_rows.iterrows():
        line = " ".join([str(v).strip() for v in r if str(v).strip() != ""]).strip()
        if line:
            notes.append(strip_html_tokens(line))
    return notes


# Excel & CSV ingestion
# ---------------------
def ingest_excel_file(path: str) -> Dict[str, Any]:
    xls = pd.ExcelFile(path)
    sheets = {}
    for sheet in xls.sheet_names:
        try:
            raw = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=str)
        except Exception:
            sheets[sheet] = {"table": pd.DataFrame(), "detected_header_row": None, "notes": [], "raw_preview": pd.DataFrame()}
            continue

        if raw is None or raw.empty:
            sheets[sheet] = {"table": pd.DataFrame(), "detected_header_row": None, "notes": [], "raw_preview": pd.DataFrame()}
            continue

        raw_preview = raw.head(12).applymap(strip_html_tokens)

        hdr_idx, combined = collapse_multirow_header(raw)
        if hdr_idx is not None and combined is not None:
            headers = sanitize_headers(combined)
            df = raw.copy()
            df.columns = headers
            df = df[hdr_idx+1:].reset_index(drop=True)
            detected_header_row = hdr_idx
        else:
            detected_header_row = detect_header_row(raw)
            if detected_header_row is not None:
                headers = sanitize_headers(raw.iloc[detected_header_row].tolist())
                df = raw.copy()
                df.columns = headers
                df = df[detected_header_row+1:].reset_index(drop=True)
            else:
                try:
                    df = pd.read_excel(xls, sheet_name=sheet, header=0, dtype=str)
                    df.columns = sanitize_headers(df.columns.tolist())
                    detected_header_row = 0
                except Exception:
                    df = raw.copy()
                    df.columns = [f"col_{i}" for i in range(df.shape[1])]
                    detected_header_row = None

        df = clean_table(df)
        notes = extract_notes_below_table(raw)

        sheets[sheet] = {
            "table": df,
            "detected_header_row": detected_header_row,
            "notes": notes,
            "raw_preview": raw_preview
        }

    return sheets

def ingest_csv_file(path: str) -> Dict[str, Any]:
    try:
        df = pd.read_csv(path, dtype=str)
        df.columns = sanitize_headers(df.columns.tolist())
        df = clean_table(df)
        raw_preview = df.head(12).astype(str).applymap(strip_html_tokens)
        return {"__csv__": {"table": df, "detected_header_row": 0, "notes": [], "raw_preview": raw_preview}}
    except Exception as e:
        return {"__csv__": {"table": pd.DataFrame(), "detected_header_row": None, "notes": [str(e)], "raw_preview": pd.DataFrame()}}


# Directory ingestion
# -------------------
def ingest_directory(directory: str = ".") -> Dict[str, Any]:
    files_in_dir = [f for f in os.listdir(directory) if f.lower().endswith((".xlsx",".xls",".csv"))]
    ingested = {}
    for fname in sorted(files_in_dir):
        path = os.path.join(directory, fname)
        try:
            if fname.lower().endswith((".xlsx",".xls")):
                sheets = ingest_excel_file(path)
            else:
                sheets = ingest_csv_file(path)
            ingested[fname] = sheets
        except Exception as e:
            ingested[fname] = {"__error__": str(e)}
    return ingested


# Completeness report
# -------------------
def completeness_report(ingested: Dict[str, Any]) -> Dict[str, Any]:
    report = {}
    for fname, sheets in ingested.items():
        report[fname] = {}
        for sname, obj in sheets.items():
            tbl = obj.get("table")
            report[fname][sname] = {
                "rows": len(tbl) if isinstance(tbl, pd.DataFrame) else 0,
                "cols": list(tbl.columns) if isinstance(tbl, pd.DataFrame) else [],
                "notes": obj.get("notes", [])
            }
    return report


# Save artifacts
# --------------
def save_ingestion_artifacts(ingested, summary, report, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    pkl_path = os.path.join(out_dir, "ingested.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(ingested, f)
    paths["ingested.pkl"] = pkl_path

    summary_path = os.path.join(out_dir, "ingested_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    paths["ingested_summary.json"] = summary_path

    report_path = os.path.join(out_dir, "completeness_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    paths["completeness_report.json"] = report_path

    return paths


# Summary printer
# ---------------
def print_ingested_summary(ingested: Dict[str, Any]) -> Dict[str, Any]:
    summary = {}
    for fname, sheets in ingested.items():
        summary[fname] = {}
        for sname, obj in sheets.items():
            tbl = obj.get("table")
            summary[fname][sname] = {
                "rows": len(tbl) if isinstance(tbl, pd.DataFrame) else 0,
                "cols": list(tbl.columns)[:20] if isinstance(tbl, pd.DataFrame) else [],
                "detected_header_row": obj.get("detected_header_row"),
                "notes_preview": (obj.get("notes") or [])[:5]
            }
    return summary
