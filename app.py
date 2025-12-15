import streamlit as st
import pandas as pd
from transformers import pipeline
from io import BytesIO

# Import agents
from agents.ingestion_agent import IngestionAgent
from agents.normalization_agent import NormalizationAgent
from agents.constraint_solver_agent import ConstraintSolverAgent
from agents.compliance_agent import ComplianceAgent
from agents.roster_generation_agent import RosterGenerationAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.explanation_agent import ExplanationAgent

# Helper: download DataFrame
def df_download_button(df, label, filename):
    if df is None or df.empty:
        st.info(f"No data to download for {label}.")
        return
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    st.download_button(
        label=f"⬇️ Download {label}",
        data=buffer.getvalue(),
        file_name=filename,
        mime="text/csv",
        key=filename,
        help=f"Download {label} as CSV"
    )

# Page setup
st.set_page_config(page_title="SchedulAI Demo", layout="wide")
st.title("📊 SchedulAI: AI‑Powered Rostering Demo")

st.markdown("""
<style>
    .success-box {background-color:#d4edda;padding:10px;border-radius:5px;color:#155724;}
    .agent-title {color:#0d6efd;font-size:22px;font-weight:bold;margin-top:20px;}
</style>
""", unsafe_allow_html=True)

# Session state
if "context" not in st.session_state:
    st.session_state.context = {}
context = st.session_state.context

# Agent 1: Ingestion
st.markdown('<div class="agent-title">🟢 Agent 1: Ingestion</div>', unsafe_allow_html=True)
if st.button("Run ingestion"):
    context = IngestionAgent().run(context)
    st.markdown('<div class="success-box">✅ Ingestion complete</div>', unsafe_allow_html=True)
    for label, df in context.items():
        if isinstance(df, pd.DataFrame):
            st.subheader(label.replace("_", " ").title())
            st.dataframe(df, use_container_width=True, height=400)
            df_download_button(df, label, f"{label}.csv")

# Agent 2: Normalization
st.markdown('<div class="agent-title">🧪 Agent 2: Normalization</div>', unsafe_allow_html=True)
if st.button("Run normalization"):
    context = NormalizationAgent().run(context)
    st.markdown('<div class="success-box">✅ Normalization complete</div>', unsafe_allow_html=True)
    st.dataframe(context["availability_tidy"], use_container_width=True, height=400)
    df_download_button(context["availability_tidy"], "Availability tidy", "availability_tidy.csv")
    st.json({
        "MIN_SHIFT_HOURS": context["MIN_SHIFT_HOURS"],
        "MIN_REST_HOURS": context["MIN_REST_HOURS"],
        "MAX_CONSEC_DAYS": context["MAX_CONSEC_DAYS"]
    })

# Agent 3: Constraint Solver
st.markdown('<div class="agent-title">🟡 Agent 3: Constraint Solver</div>', unsafe_allow_html=True)
if st.button("Run solver"):
    context = ConstraintSolverAgent().run(context)
    st.markdown('<div class="success-box">✅ Draft roster generated</div>', unsafe_allow_html=True)
    st.dataframe(context["draft_roster_df"], use_container_width=True, height=400)
    df_download_button(context["draft_roster_df"], "Draft roster", "draft_roster.csv")

# Agent 4: Compliance
st.markdown('<div class="agent-title">🔴 Agent 4: Compliance</div>', unsafe_allow_html=True)
if st.button("Run compliance"):
    context = ComplianceAgent().run(context)
    st.markdown('<div class="success-box">⚠️ Compliance check complete</div>', unsafe_allow_html=True)
    st.dataframe(context["compliance_violations"], use_container_width=True, height=300)
    df_download_button(context["compliance_violations"], "Compliance violations", "compliance.csv")

# Agent 5: Roster Generation
st.markdown('<div class="agent-title">🔵 Agent 5: Roster Generation</div>', unsafe_allow_html=True)
if st.button("Assemble final roster"):
    context = RosterGenerationAgent().run(context)
    st.markdown('<div class="success-box">✅ Final roster assembled</div>', unsafe_allow_html=True)
    st.dataframe(context["final_roster_df"], use_container_width=True, height=400)
    df_download_button(context["final_roster_df"], "Final roster", "final_roster.csv")

    # Pivot view
    pivot = context["final_roster_df"].pivot_table(
        index="day_label",
        columns="employee_name",
        values="shift_code",
        aggfunc=lambda x: ";".join(str(v) for v in x if pd.notnull(v))
    )
    st.subheader("📅 Employee Schedule Grid (2-week overview)")
    st.dataframe(pivot, use_container_width=True, height=400)
    df_download_button(pivot, "Schedule grid", "schedule_grid.csv")

# Agent 6: Knowledge
st.markdown('<div class="agent-title">🧠 Agent 6: Knowledge</div>', unsafe_allow_html=True)
if st.button("Run knowledge agent"):
    context = KnowledgeAgent().run(context)
    st.markdown('<div class="success-box">✅ Knowledge agent complete</div>', unsafe_allow_html=True)
    st.info("You can now query compliance knowledge below.")
    query = st.text_input("Ask a compliance question:")
    if query:
        results = context["rag_query_fn"](query)
        for doc, meta in results:
            st.write(f"Source: {meta['source']}")
            st.text(doc[:300] + "...")

# Agent 7: Explanation
st.markdown('<div class="agent-title">🟣 Agent 7: Explanation</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")

if st.button("Generate explanations"):
    llm = load_model()

    def llm_wrapper(prompt: str) -> str:
        try:
            return llm(prompt, max_length=150)[0]["generated_text"]
        except Exception as e:
            return f"LLM error: {e}"

    context = ExplanationAgent(llm=llm_wrapper).run(context)
    st.markdown('<div class="success-box">✅ Explanations generated</div>', unsafe_allow_html=True)
    st.dataframe(context["explanations_df"], use_container_width=True, height=500)
    df_download_button(context["explanations_df"], "Explanations", "explanations.csv")

    # Optional search/filter
    st.subheader("🔍 Search explanations")
    emp_query = st.text_input("Enter employee name or day to filter:")
    if emp_query:
        filtered = context["explanations_df"][
            context["explanations_df"].apply(lambda row: emp_query.lower() in str(row).lower(), axis=1)
        ]
        st.dataframe(filtered, use_container_width=True, height=400)
        df_download_button(filtered, "Filtered explanations", "filtered_explanations.csv")
