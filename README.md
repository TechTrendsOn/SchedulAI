# 📊 SchedulAI: AI‑Powered Restaurant Rostering

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://schedulai-demo.streamlit.app)

SchedulAI is a modular, multi‑agent rostering system built with Streamlit. It automates staff scheduling while enforcing compliance, operational rules, and manager‑friendly explanations.

---

## 🚀 Features

- **Agent‑based pipeline**
  - **Ingestion Agent**: Loads and cleans Excel/CSV inputs (parameters, availability, store configs).
  - **Normalization Agent**: Tidy transformation of availability and extraction of canonical constraints.
  - **Constraint Solver Agent**: OR‑Tools CP‑SAT to generate draft rosters with shift/rest/demand constraints.
  - **Compliance Agent**: Flags roster rule violations and annotates outcomes.
  - **Roster Generation Agent**: Produces final rosters with pivoted schedule grids and notes.
  - **Knowledge Agent (RAG)**: ChromaDB store for compliance rules and parameters; supports grounded queries.
  - **Explanation Agent (optional)**: Hugging Face models for manager‑friendly rationales.

- **Streamlit UI**
  - Sectioned workflow, success boxes, and scrollable tables.
  - CSV download buttons at each step.
  - Queryable compliance knowledge and searchable rosters.

---

## 📂 Project structure


---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/TechTrendsOn/SchedulAI.git
cd SchedulAI
pip install -r requirements.txt

Run 
