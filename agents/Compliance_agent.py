# Compliance Agent (Validation & Rule Checking)
import pandas as pd

class ComplianceAgent:
    def run(self, context):
        roster_df = context.get("draft_roster_df")
        basic_params = context.get("basic_params")
        compliance_notes = context.get("compliance_notes")
        service_periods = context.get("service_periods")
        store_configs = context.get("store_configs")

        violations = []

        # Numeric checks (already enforced in solver, but double-check here)
        MIN_SHIFT_HOURS = context["MIN_SHIFT_HOURS"]
        MIN_REST_HOURS = context["MIN_REST_HOURS"]
        MAX_CONSEC_DAYS = context["MAX_CONSEC_DAYS"]

        # Check minimum shift length
        for _, row in roster_df.iterrows():
            if pd.notnull(row["start_time"]) and pd.notnull(row["end_time"]):
                duration = row["end_time"] - row["start_time"]
                if duration < MIN_SHIFT_HOURS:
                    violations.append({
                        "employee": row["employee_name"],
                        "day": row["day_label"],
                        "issue": f"Shift shorter than {MIN_SHIFT_HOURS} hours"
                    })

        # Check rest periods between consecutive shifts
        for emp_id, group in roster_df.groupby("employee_id"):
            group_sorted = group.sort_values("day_label")
            prev_end = None
            for _, row in group_sorted.iterrows():
                if prev_end is not None and row["start_time"] is not None:
                    rest = row["start_time"] - prev_end
                    if rest < MIN_REST_HOURS:
                        violations.append({
                            "employee": row["employee_name"],
                            "day": row["day_label"],
                            "issue": f"Rest period shorter than {MIN_REST_HOURS} hours"
                        })
                prev_end = row["end_time"]

        # Check maximum consecutive days
        for emp_id, group in roster_df.groupby("employee_id"):
            days = list(group["day_label"].unique())
            if len(days) > MAX_CONSEC_DAYS:
                violations.append({
                    "employee": group["employee_name"].iloc[0],
                    "issue": f"Scheduled for more than {MAX_CONSEC_DAYS} consecutive days"
                })

        # Textual compliance checks (using notes/configs)
        # Example: check if store configs mention "Closed" for a day but roster has shifts
        if store_configs is not None and "Day" in store_configs.columns:
            for _, cfg in store_configs.iterrows():
                if "Closed" in str(cfg.get("Status", "")):
                    day = cfg["Day"]
                    if day in roster_df["day_label"].values:
                        violations.append({
                            "day": day,
                            "issue": "Rostered staff on a closed store day"
                        })

        # Example: check compliance notes for keywords like "meal break"
        if compliance_notes is not None:
            notes_text = " ".join(compliance_notes.astype(str).values.flatten())
            if "meal break" in notes_text.lower():
                # Simple heuristic: flag shifts longer than 5 hours without explicit break
                for _, row in roster_df.iterrows():
                    if pd.notnull(row["start_time"]) and pd.notnull(row["end_time"]):
                        duration = row["end_time"] - row["start_time"]
                        if duration > 5:  # hours
                            violations.append({
                                "employee": row["employee_name"],
                                "day": row["day_label"],
                                "issue": "Shift >5 hours without meal break (per compliance notes)"
                            })

        # Update context
        context["compliance_violations"] = pd.DataFrame(violations)
        context["compliance_passed"] = len(violations) == 0

        return context
