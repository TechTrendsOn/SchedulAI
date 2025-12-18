# app/streamlit_app.py
import os
import json
import glob
import pandas as pd
import altair as alt
import streamlit as st
from dotenv import load_dotenv

# Load Hugging Face token from .env
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Import agents (assumes your repo structure matches earlier messages)
from agents.ingestion_agent import ingest_directory, print_ingested_summary, completeness_report, save_ingestion_artifacts
from agents.preprocessor_agent import PreprocessorAgent
from agents.parameters_agent import ParametersAgent
from agents.normalization_agent import NormalizationAgent
from agents.solver_agent import SolverAgent
from agents.compliance_agent import ComplianceAgent
from agents.final_roster_agent import FinalRosterAgent
from agents.explanation_agent import ExplanationAgent
from agents.knowledge_agent import KnowledgeAgentRAG

# Helpers
def get_latest_confirmed_report(pattern: str):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No confirmed compliance reports found.")
    return max(files, key=os.path.getmtime)

DATA_IN = "data/inputs"
DATA_OUT = "data/artifacts"
os.makedirs(DATA_IN, exist_ok=True)
os.makedirs(DATA_OUT, exist_ok=True)

st.set_page_config(page_title="Roster Compliance Demo", layout="wide")
st.title("Roster Compliance Demo — End-to-End Agents")

page = st.sidebar.radio("Select Agent/Page", [
    "1. Ingestion",
    "2. Preprocessor",
    "3. Parameters",
    "4. Normalization",
    "5. Solver",
    "6. Compliance",
    "7. Final Roster",
    "8. Knowledge Agent Chat",
    "9. Explanation Agent Chat"
])

# -----------------------------------------------------------------------------
# 1. Ingestion
# -----------------------------------------------------------------------------
if page.startswith("1"):
    st.header("Ingestion Agent")
    st.write("Upload Excel/CSV files into data/inputs and run ingestion. Artifacts and audit JSONs are saved in data/artifacts.")

    uploaded = st.file_uploader("Upload Excel/CSV files", type=["csv","xlsx","xls"], accept_multiple_files=True)
    if uploaded:
        for file in uploaded:
            path = os.path.join(DATA_IN, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"Saved {len(uploaded)} file(s) to {DATA_IN}")

    if st.button("Run Ingestion"):
        ingested = ingest_directory(DATA_IN)
        summary = print_ingested_summary(ingested)
        report = completeness_report(ingested)
        paths = save_ingestion_artifacts(ingested, summary, report, DATA_OUT)

        st.subheader("Completeness report")
        st.json(report)

        st.subheader("Files summary")
        st.json(summary)

        st.subheader("Downloads")
        for label, path in paths.items():
            with open(path, "rb") as f:
                st.download_button(f"Download {os.path.basename(path)}", f.read(), file_name=os.path.basename(path))

# -----------------------------------------------------------------------------
# 2. Preprocessor
# -----------------------------------------------------------------------------
elif page.startswith("2"):
    st.header("Preprocessor Agent")
    st.write("Normalize availability, shift codes, and fixed hours. Produces an audit manifest.")

    ingested_path = st.text_input("Ingested pickle path", f"{DATA_OUT}/ingested.pkl")
    out_dir = st.text_input("Output dir", DATA_OUT)

    if st.button("Run PreprocessorAgent"):
        agent = PreprocessorAgent(ingested_path=ingested_path, out_dir=out_dir)
        manifest = agent.run()
        st.success("Preprocessor finished.")
        st.subheader("Manifest")
        st.json(manifest)

        # Testing: previews and downloads
        for k in ("availability_cleaned", "shift_codes_cleaned", "fixed_hours_tidy"):
            path = manifest["outputs"].get(k)
            if path and os.path.exists(path):
                st.subheader(k)
                df = pd.read_csv(path, dtype=str)
                st.dataframe(df.head(50), use_container_width=True)
                with open(path, "rb") as f:
                    st.download_button(f"Download {os.path.basename(path)}", f.read(), file_name=os.path.basename(path))

# -----------------------------------------------------------------------------
# 3. Parameters
# -----------------------------------------------------------------------------
elif page.startswith("3"):
    st.header("Parameters Agent")
    st.write("Extract rostering parameters (meal breaks, penalty rates, compliance notes).")

    ingested_path = st.text_input("Ingested pickle path", f"{DATA_OUT}/ingested.pkl")
    out_dir = st.text_input("Output dir", DATA_OUT)

    if st.button("Run ParametersAgent"):
        agent = ParametersAgent(ingested_path=ingested_path, out_dir=out_dir)
        manifest = agent.run()
        st.success("ParametersAgent finished.")
        st.subheader("Manifest")
        st.json(manifest)

        params_path = manifest["outputs"].get("rostering_parameters")
        if params_path and os.path.exists(params_path):
            st.subheader("Extracted parameters")
            params = json.load(open(params_path))
            st.json(params)
            with open(params_path, "rb") as f:
                st.download_button("Download rostering_parameters.json", f.read(), file_name="rostering_parameters.json")

        # Testing: show concise sheet summaries if ingested.pkl is available
        if os.path.exists(ingested_path):
            import pickle
            with open(ingested_path, "rb") as f:
                ingested = pickle.load(f)
            st.subheader("Sheet summaries")
            for fname, sheets in ingested.items():
                st.markdown(f"**File:** {fname}")
                if isinstance(sheets, dict):
                    for sname, obj in sheets.items():
                        tbl = obj.get("table")
                        if isinstance(tbl, pd.DataFrame):
                            st.write(f"Sheet: {sname}, rows={len(tbl)}, cols={list(tbl.columns)[:6]}")

# -----------------------------------------------------------------------------
# 4. Normalization
# -----------------------------------------------------------------------------
elif page.startswith("4"):
    st.header("Normalization Agent")
    st.write("Normalize availability tokens for consistency.")

    availability_path = st.text_input("Availability cleaned CSV", f"{DATA_OUT}/availability_cleaned_for_norm.csv")
    shift_codes_path = st.text_input("Shift codes cleaned CSV", f"{DATA_OUT}/shift_codes_cleaned.csv")
    out_dir = st.text_input("Output dir", DATA_OUT)

    if st.button("Run NormalizationAgent"):
        agent = NormalizationAgent(availability_path=availability_path, shift_codes_path=shift_codes_path, out_dir=out_dir)
        manifest = agent.run()
        st.success("Normalization finished.")
        st.subheader("Manifest")
        st.json(manifest)

        norm_path = manifest["outputs"].get("availability_normalized")
        if norm_path and os.path.exists(norm_path):
            st.subheader("Normalized availability preview")
            df = pd.read_csv(norm_path, dtype=str)
            st.dataframe(df.head(50), use_container_width=True)
            with open(norm_path, "rb") as f:
                st.download_button("Download availability_normalized.csv", f.read(), file_name="availability_normalized.csv")

# -----------------------------------------------------------------------------
# 5. Solver
# -----------------------------------------------------------------------------
elif page.startswith("5"):
    st.header("Solver Agent")
    st.write("Generate draft roster using CP-SAT solver.")

    avail = st.text_input("Availability tidy CSV", f"{DATA_OUT}/availability_tidy.csv")
    shifts = st.text_input("Shift codes cleaned CSV", f"{DATA_OUT}/shift_codes_cleaned.csv")
    fixed = st.text_input("Fixed hours tidy CSV", f"{DATA_OUT}/fixed_hours_tidy.csv")
    params = st.text_input("Parameters JSON", f"{DATA_OUT}/rostering_parameters.json")

    if st.button("Run SolverAgent"):
        agent = SolverAgent(avail, shifts, fixed, params, out_dir=DATA_OUT, time_limit_sec=120, num_workers=8)
        status = agent.run()
        st.success(f"Solver finished with status {status}")

        roster_path = os.path.join(DATA_OUT, "roster_solution.csv")
        if os.path.exists(roster_path):
            df = pd.read_csv(roster_path)
            st.subheader("Draft roster preview")
            st.dataframe(df.head(50), use_container_width=True)
            with open(roster_path, "rb") as f:
                st.download_button("Download roster_solution.csv", f.read(), file_name="roster_solution.csv")

# -----------------------------------------------------------------------------
# 6. Compliance
# -----------------------------------------------------------------------------
elif page.startswith("6"):
    st.header("Compliance Agent")
    st.write("Run compliance checks and produce a structured report.")

    roster_path = st.text_input("Draft roster CSV", f"{DATA_OUT}/roster_solution.csv")
    params_path = st.text_input("Parameters JSON", f"{DATA_OUT}/rostering_parameters.json")
    out_dir = st.text_input("Output dir", DATA_OUT)
    fallback_year = st.number_input("Fallback year", value=2024, step=1)

    if st.button("Run ComplianceAgent"):
        comp_agent = ComplianceAgent(roster_path=roster_path, params_path=params_path, out_dir=out_dir, fallback_year=fallback_year)
        report_path = comp_agent.run()
        st.success("Compliance report generated.")
        report = json.load(open(report_path))
        st.subheader("Compliance report JSON")
        st.json(report)

        with open(report_path, "rb") as f:
            st.download_button("Download compliance_report.json", f.read(), file_name="compliance_report.json")

        # Testing i): recompute key counts to verify JSON vs raw data
        st.subheader("Validation: recompute key violations from roster_solution.csv")
        df = pd.read_csv(roster_path)
        df["hours"] = pd.to_numeric(df["hours"], errors="coerce").fillna(0.0)

        from datetime import datetime

        def parse_date(date_str, fallback_year):
            try:
                base = datetime.strptime(str(date_str).strip(), "%a %b %d")
                return base.replace(year=fallback_year)
            except Exception:
                return None

        def parse_time(time_str):
            try:
                hh, mm = map(int, str(time_str).split(":"))
                return hh, mm
            except Exception:
                return None

        df["date_parsed"] = df["date"].apply(lambda d: parse_date(d, fallback_year))
        df["start_dt"] = df.apply(
            lambda r: r["date_parsed"].replace(hour=parse_time(r["start_time"])[0], minute=parse_time(r["start_time"])[1])
            if r["date_parsed"] and parse_time(r["start_time"]) else None, axis=1)
        df["end_dt"] = df.apply(
            lambda r: r["date_parsed"].replace(hour=parse_time(r["end_time"])[0], minute=parse_time(r["end_time"])[1])
            if r["date_parsed"] and parse_time(r["end_time"]) else None, axis=1)

        # Rest period violations (adjacent)
        rest_violations = 0
        for emp, group in df.groupby("employee"):
            g = group.sort_values(["date_parsed", "start_dt"])
            prev_end = None
            for _, row in g.iterrows():
                if prev_end and row["start_dt"]:
                    rest_hours = (row["start_dt"] - prev_end).total_seconds() / 3600.0
                    if rest_hours < 10:
                        rest_violations += 1
                if row["end_dt"]:
                    prev_end = row["end_dt"]

        # Meal break violations (> 5h)
        meal_violations = (df["hours"] > 5).sum()

        # Weekly hours band violations
        bands = {"full-time": (35, 38), "part-time": (20, 32), "casual": (8, 24)}
        df["iso_week"] = df["date_parsed"].dt.isocalendar().week
        weekly_violations = 0
        for (emp, week), group in df.groupby(["employee", "iso_week"]):
            typ = str(group.iloc[0].get("type", "")).lower()
            if "full" in typ:
                band = bands["full-time"]
            elif "part" in typ:
                band = bands["part-time"]
            elif "casual" in typ:
                band = bands["casual"]
            else:
                continue
            total_hours = group["hours"].sum()
            if total_hours < band[0] or total_hours > band[1]:
                weekly_violations += 1

        st.write(f"Rest Period violations: {rest_violations}")
        st.write(f"Meal Break violations: {meal_violations}")
        st.write(f"Weekly Hours violations: {weekly_violations}")
        st.write(f"Total violations: {rest_violations + meal_violations + weekly_violations}")

        # Testing ii): display the JSON report directly
        st.subheader("Raw compliance_report.json")
        st.code(json.dumps(report, indent=2))

# -----------------------------
# 6b. Compliance & Swaps (interactive confirmation)
# -----------------------------
elif page.startswith("6b"):
    st.header("Compliance & Swaps — Interactive Confirmation")
    raw_compliance_path = f"{DATA_OUT}/compliance_report.json"

    if os.path.exists(raw_compliance_path):
        with open(raw_compliance_path, "r") as f:
            compliance = json.load(f)

        violations = compliance.get("violations", [])
        swaps = compliance.get("swaps", [])

        # Build actionable items: each violation + its suggested swap (if any)
        actionable_items = []
        for v in violations:
            match = next((s for s in swaps if s.get("violation") == v), None)
            actionable_items.append({
                "violation": v,
                "swap": match,
                "confirmed": False
            })

        if actionable_items:
            st.subheader("Review & Confirm Fixes")

            for idx, item in enumerate(actionable_items):
                v = item["violation"]
                s = item["swap"]

                with st.expander(f"Issue {idx+1}: {v.get('type')} — {v.get('employee')} on {v.get('date')}"):
                    st.write("### Violation")
                    st.json(v)

                    if s:
                        st.write("### Suggested Swap")
                        st.json(s)
                        label = f"Confirm swap: {v.get('employee')} → {s.get('suggested_employee')}"
                    else:
                        label = "Acknowledge this violation"

                    confirm = st.checkbox(label, key=f"confirm_{idx}")
                    item["confirmed"] = bool(confirm)

            if st.button("Save confirmed compliance report"):
                import datetime
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                confirmed_path = os.path.join(DATA_OUT, f"compliance_report_confirmed_{ts}.json")

                with open(confirmed_path, "w") as f:
                    json.dump({"confirmed_items": actionable_items}, f, indent=2)

                st.success(f"Saved: {confirmed_path}")

        else:
            st.info("No violations found in compliance report.")

    else:
        st.warning("Run ComplianceAgent first to generate compliance_report.json.")



# -----------------------------------------------------------------------------
# 7. Final Roster
# -----------------------------------------------------------------------------
elif page.startswith("7"):
    st.header("Final Roster Agent")
    st.write("Merge confirmed compliance fixes and swaps, then produce final roster and manifest.")

    roster_path = st.text_input("Draft roster CSV", f"{DATA_OUT}/roster_solution.csv")
    manifest_path = st.text_input("Parameters manifest JSON", f"{DATA_OUT}/parameters_manifest.json")
    use_latest_confirmed = st.checkbox("Auto-detect latest confirmed compliance report", value=True)
    if use_latest_confirmed:
        try:
            confirmed_path = get_latest_confirmed_report(pattern=f"{DATA_OUT}/compliance_report_confirmed_*.json")
            st.caption(f"Confirmed compliance file: {confirmed_path}")
        except FileNotFoundError:
            confirmed_path = None
            st.warning("No confirmed compliance reports found. Save one from Compliance & Swaps page.")
    else:
        confirmed_path = st.text_input("Confirmed compliance path", f"{DATA_OUT}/compliance_report_confirmed_YYYYMMDD_HHMMSS.json")

    if st.button("Generate Final Roster"):
        agent = FinalRosterAgent(roster_path=roster_path, compliance_path=confirmed_path, manifest_path=manifest_path)
        agent.apply_compliance()
        final_csv, final_manifest = agent.generate_final(out_csv=f"{DATA_OUT}/final_roster.csv", out_json=f"{DATA_OUT}/final_roster_manifest.json")

        st.subheader("Compliance summary")
        st.text(agent.summary_report())

        st.subheader("Final roster preview")
        final_df = pd.read_csv(final_csv)
        st.dataframe(final_df.head(50), use_container_width=True)

        # Visual: Draft vs Final by shift counts
        if os.path.exists(roster_path):
            draft_df = pd.read_csv(roster_path)
            if {"employee", "shift"}.issubset(draft_df.columns) and {"employee", "shift"}.issubset(final_df.columns):
                draft_counts = draft_df.groupby("shift")["employee"].count().reset_index().rename(columns={"employee": "draft_count"})
                final_counts = final_df.groupby("shift")["employee"].count().reset_index().rename(columns={"employee": "final_count"})
                compare = pd.merge(draft_counts, final_counts, on="shift", how="outer").fillna(0)
                chart = alt.Chart(compare).transform_fold(["draft_count", "final_count"]).mark_bar().encode(
                    x="shift:N", y="value:Q", color="key:N"
                ).properties(height=300)
                st.subheader("Draft vs Final — counts per shift")
                st.altair_chart(chart, use_container_width=True)

        st.subheader("Downloads")
        for path in (final_csv, final_manifest):
            with open(path, "rb") as f:
                st.download_button(f"Download {os.path.basename(path)}", f.read(), file_name=os.path.basename(path))

# -----------------------------------------------------------------------------
# 8. Knowledge Agent Chat
# -----------------------------------------------------------------------------
elif page.startswith("8"):
    st.header("Knowledge Agent Chat")
    st.write("Ask grounded compliance questions from the ingested knowledge base.")

    if st.button("Load KnowledgeAgentRAG"):
        agent = KnowledgeAgentRAG(
            persistence_dir="./chroma_knowledge",
            collection_name="compliance_knowledge",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            llm_model_name="google/flan-t5-base",
            chunk_size=1200,
            chunk_overlap=200
        )
        st.session_state["knowledge_agent"] = agent
        st.success("KnowledgeAgentRAG loaded.")

    # Optional ingestion trigger (you can comment this if already ingested)
    st.subheader("Optional: Ingest files")
    ingest_click = st.button("Ingest default files")
    if ingest_click and "knowledge_agent" in st.session_state:
        agent = st.session_state["knowledge_agent"]
        agent.ingest_files([
            {"path": os.path.join(DATA_IN, "australian_restaurant_rostering_parameters.xlsx"), "type": "excel"},
            {"path": os.path.join(DATA_OUT, "rostering_parameters.json"), "type": "json"},
            {"path": os.path.join(DATA_OUT, "parameters_manifest.json"), "type": "json"},
            {"path": os.path.join(DATA_IN, "employee_availability_2weeks.xlsx"), "type": "excel"},
            {"path": os.path.join(DATA_IN, "fixed_hours_template_2columns.xlsx"), "type": "excel"},
            {"path": os.path.join(DATA_IN, "store_configurations.xlsx"), "type": "excel"},
            {"path": os.path.join(DATA_IN, "management_roster_simplified.xlsx"), "type": "excel"},
            {"path": os.path.join(DATA_OUT, "compliance_report.json"), "type": "json"},
        ])
        st.success("Knowledge files ingested.")

    if "knowledge_agent" in st.session_state:
        query = st.text_area("Ask a compliance knowledge question", "What is the minimum hours between shifts?")
        if st.button("Answer"):
            agent = st.session_state["knowledge_agent"]
            res = agent.answer(query)
            st.subheader("Answer")
            st.json(res)

# -----------------------------------------------------------------------------
# 9. Explanation Agent Chat
# -----------------------------------------------------------------------------
elif page.startswith("9"):
    st.header("Explanation Agent Chat")
    st.write("Generate a narrative, auditor-friendly explanation based on compliance report and final roster manifest.")

    compliance_path = st.text_input("Confirmed compliance file", f"{DATA_OUT}/compliance_report_confirmed_YYYYMMDD_HHMMSS.json")
    manifest_path = st.text_input("Final roster manifest", f"{DATA_OUT}/final_roster_manifest.json")
    rules_path = st.text_input("Compliance rules (optional)", f"{DATA_IN}/compliance_rules.json")

    if st.button("Load ExplanationAgent"):
        st.session_state["exp_agent"] = ExplanationAgent(compliance_path, manifest_path, rules_path if os.path.exists(rules_path) else None)
        st.success("ExplanationAgent loaded.")

    if "exp_agent" in st.session_state:
        query = st.text_area("Ask a compliance question (optional)", "Why was Alice’s shift swapped?")
        if st.button("Generate explanation"):
            agent = st.session_state["exp_agent"]
            # Generate the human-readable explanation
            narrative = agent.human_report()
            st.subheader("Explanation")
            st.text_area("Agent narrative", narrative, height=300)

            # Save and download the explanation report
            out_json = os.path.join(DATA_OUT, "explanation_report.json")
            explanation = agent.generate_explanation(out_json)
            with open(out_json, "rb") as f:
                st.download_button("Download explanation_report.json", f.read(), file_name="explanation_report.json")

            # Testing: show final roster manifest and compliance report excerpts
            st.subheader("Final roster manifest excerpt")
            if os.path.exists(manifest_path):
                st.code(json.dumps(json.load(open(manifest_path)), indent=2)[:2000])
            st.subheader("Compliance report excerpt")
            if os.path.exists(compliance_path):
                st.code(json.dumps(json.load(open(compliance_path)), indent=2)[:2000])
