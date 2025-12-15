# Roster Generation Agent (Final Assembly)
import pandas as pd

class RosterGenerationAgent:
    def run(self, context):
        draft_roster = context.get("draft_roster_df")
        compliance_passed = context.get("compliance_passed", False)
        violations = context.get("compliance_violations")

        final_roster = draft_roster.copy()

        # If compliance failed, annotate roster with violation notes
        if not compliance_passed and violations is not None and not violations.empty:
            # Merge violations into roster for visibility
            violations_summary = violations.groupby("employee")["issue"].apply(lambda x: "; ".join(x)).reset_index()
            final_roster = final_roster.merge(violations_summary, left_on="employee_name", right_on="employee", how="left")
            final_roster.rename(columns={"issue": "compliance_notes"}, inplace=True)
        else:
            final_roster["compliance_notes"] = "All checks passed"

        # Update context with final roster
        context["final_roster_df"] = final_roster
        return context

