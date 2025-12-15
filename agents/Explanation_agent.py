# Explanation Agent (Manager-friendly, RAG-grounded)
import pandas as pd

class ExplanationAgent:
    def __init__(self, llm=None, max_snippets=3, include_sources=True):
        """
        llm: optional callable like llm(prompt:str)->str
             If None, uses a deterministic template.
        max_snippets: number of RAG documents to include for grounding.
        include_sources: whether to include source tags in the output.
        """
        self.llm = llm
        self.max_snippets = max_snippets
        self.include_sources = include_sources

    # Helpers
    def _format_time(self, t):
        """Format decimal hours (e.g., 6.5) as HH:MM; returns '-' if None/NaN."""
        if t is None or pd.isna(t):
            return "-"
        h = int(t)
        m = int(round((t - h) * 60))
        return f"{h:02d}:{m:02d}"

    def _collect_rag(self, context, row):
        """Query RAG for relevant rule snippets using compact, meaningful terms."""
        rag_query = context.get("rag_query_fn", None)
        if not rag_query:
            return []
        query_terms = ["minimum rest hours", "minimum hours per shift", "consecutive days", "meal break"]
        if pd.notnull(row.get("shift_code")):
            query_terms.append(str(row["shift_code"]))
        if pd.notnull(row.get("station")):
            query_terms.append(str(row["station"]))
        query = " | ".join([t for t in query_terms if t])
        return rag_query(query, n_results=self.max_snippets)

    def _summarize_row_violations(self, violations_df, row):
        """Return a single-line summary of violations for the given row."""
        if violations_df is None or violations_df.empty:
            return ""
        mask = False
        if {"employee", "day"}.issubset(violations_df.columns):
            mask = (violations_df["employee"] == row.get("employee_name")) & \
                   (violations_df["day"] == row.get("day_label"))
        elif "employee" in violations_df.columns:
            mask = (violations_df["employee"] == row.get("employee_name"))
        else:
            return ""
        subset = violations_df[mask]
        if subset.empty or "issue" not in subset.columns:
            return ""
        issues = sorted(set(subset["issue"].astype(str).tolist()))
        return "; ".join(issues)

    def build_prompt(self, row, rag_snippets, violation_summary, params):
        """Construct a concise, grounded prompt for LLM generation."""
        start_s = self._format_time(row.get("start_time"))
        end_s = self._format_time(row.get("end_time"))
        rules_text = "\n".join([f"- {m.get('source','unknown')}: {doc[:300]}" for doc, m in rag_snippets]) \
                     or "- No specific rules found."
        params_line = (
            f"Parameters: min_shift_hours={params.get('MIN_SHIFT_HOURS','?')}, "
            f"min_rest_hours={params.get('MIN_REST_HOURS','?')}, "
            f"max_consec_days={params.get('MAX_CONSEC_DAYS','?')}"
        )
        violation_line = f"Known violations: {violation_summary or 'None'}"
        return (
            f"Employee: {row.get('employee_name','N/A')} ({row.get('employment_type','N/A')}) | "
            f"Station: {row.get('station','N/A')} | Day: {row.get('day_label','N/A')}\n"
            f"Shift: {row.get('shift_code','N/A')} | {start_s}–{end_s}\n"
            f"{params_line}\n{violation_line}\n\n"
            f"Relevant rules:\n{rules_text}\n\n"
            f"Task: In one short, manager-friendly paragraph, explain why this assignment is compliant "
            f"or what needs to change. Be specific and neutral."
        )

    # Main run
    def run(self, context):
        final_roster = context.get("final_roster_df", pd.DataFrame())
        violations_df = context.get("compliance_violations", pd.DataFrame())
        params = {
            "MIN_SHIFT_HOURS": context.get("MIN_SHIFT_HOURS"),
            "MIN_REST_HOURS": context.get("MIN_REST_HOURS"),
            "MAX_CONSEC_DAYS": context.get("MAX_CONSEC_DAYS"),
        }

        explanations = []
        for _, row in final_roster.iterrows():
            rag_snippets = self._collect_rag(context, row)
            violation_summary = self._summarize_row_violations(violations_df, row)

            if self.llm:
                prompt = self.build_prompt(row, rag_snippets, violation_summary, params)
                try:
                    text = self.llm(prompt)
                except Exception:
                    text = "Explanation unavailable due to model error; falling back to template."
                    self.llm = None  # switch to template

            if not self.llm:
                start_s = self._format_time(row.get("start_time"))
                end_s = self._format_time(row.get("end_time"))
                notes = row.get("compliance_notes", "")
                sources = "; ".join(sorted(set([m.get("source","unknown") for _, m in rag_snippets]))) if self.include_sources else ""
                base = (
                    f"{row.get('employee_name','N/A')} scheduled at {row.get('station','N/A')} on {row.get('day_label','N/A')} "
                    f"{start_s}-{end_s} ({row.get('shift_code','N/A')}). "
                )
                status = (
                    f"Notes: {notes}. " if notes and notes != "All checks passed"
                    else "Compliant per current checks. "
                )
                viol = f"Violations: {violation_summary}. " if violation_summary else ""
                srcs = f"Sources: {sources}" if sources else ""
                text = base + status + viol + srcs

            explanations.append({
                "employee_id": row.get("employee_id"),
                "employee_name": row.get("employee_name"),
                "employment_type": row.get("employment_type", ""),
                "day_label": row.get("day_label"),
                "station": row.get("station"),
                "shift_code": row.get("shift_code"),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "explanation": text
            })

        explanations_df = pd.DataFrame(explanations)
        context["explanations_df"] = explanations_df
        return context
