# Roster Generation Agent (Final Assembly)
# Merges the solver’s output with confirmed compliance fixes and
# swaps to produce the authoritative, audit‑ready final roster.

import pandas as pd
import json
import os

# Safe fallback for Streamlit helper
try:
    from app.streamlit_app import get_latest_confirmed_report
except Exception:
    def get_latest_confirmed_report():
        raise RuntimeError(
            "No compliance file provided and auto-detection unavailable. "
            "Pass compliance_path explicitly."
        )

class FinalRosterAgent:
    def __init__(self, roster_path: str, compliance_path: str, manifest_path: str):
        # Load solver roster
        self.roster_df = pd.read_csv(roster_path)

        # Auto-detect latest confirmed compliance file if not provided
        if compliance_path is None:
            compliance_path = get_latest_confirmed_report()

        self.compliance_file_used = compliance_path

        # Load confirmed compliance report
        with open(compliance_path, "r") as f:
            self.compliance_report = json.load(f)

        # Load parameters manifest
        with open(manifest_path, "r") as f:
            self.parameters_manifest = json.load(f)

        # Track applied changes
        self.summary_stats = {
            "swaps_applied": 0,
            "direct_fixes": 0,
            "meal_breaks_inserted": 0,
            "penalty_rates_adjusted": 0
        }

        self.applied_swaps = []
        self.suggested_swaps = []

    
    # Apply confirmed compliance fixes and swaps
    # ------------------------------------------
    def apply_compliance(self):
        """
        Merge confirmed compliance fixes and swaps into the roster.
        Uses 'violations' and 'swaps' from compliance_report.json.
        """

        violations = self.compliance_report.get("violations", [])
        swaps = self.compliance_report.get("swaps", [])

        
        # 1. Direct fixes
        # ---------------
        for v in violations:
            emp = v.get("employee")
            rule = v.get("rule", "")

            # Add notes to the employee’s row
            self.roster_df.loc[self.roster_df["employee"] == emp, "notes"] = rule
            self.summary_stats["direct_fixes"] += 1

            if "meal break" in rule.lower():
                self.summary_stats["meal_breaks_inserted"] += 1
            if "penalty" in rule.lower():
                self.summary_stats["penalty_rates_adjusted"] += 1

        
        # 2. Apply confirmed swaps
        # ------------------------
        for s in swaps:
            if not s.get("confirmed", False):
                continue

            emp_a = s.get("employee")
            emp_b = s.get("suggested_employee")
            date = s.get("date")
            station = s.get("station")

            # Identify rows to swap
            row_a = self.roster_df[
                (self.roster_df["employee"] == emp_a) &
                (self.roster_df["date"] == date) &
                (self.roster_df["station"] == station)
            ]

            row_b = self.roster_df[
                (self.roster_df["employee"] == emp_b) &
                (self.roster_df["date"] == date) &
                (self.roster_df["station"] == station)
            ]

            if row_a.empty or row_b.empty:
                continue

            # Perform swap
            self.roster_df.loc[row_a.index, "employee"] = emp_b
            self.roster_df.loc[row_b.index, "employee"] = emp_a
            self.roster_df.loc[row_a.index.union(row_b.index), "notes"] = "Swap applied"

            self.summary_stats["swaps_applied"] += 1
            self.applied_swaps.append(s)

        # Keep all suggested swaps for manifest
        self.suggested_swaps = swaps

    
    # Save final roster and manifest
    # ------------------------------
    def generate_final(self, out_csv="final_roster.csv", out_json="final_roster_manifest.json"):
        self.roster_df.to_csv(out_csv, index=False)

        manifest = {
            "parameters_used": self.parameters_manifest,
            "compliance_file_used": self.compliance_file_used,
            "compliance_applied": self.compliance_report,
            "final_roster_rows": len(self.roster_df),
            "summary": self.summary_stats,
            "applied_swaps": self.applied_swaps,
            "suggested_swaps": self.suggested_swaps,
            "notes": "Roster merged with confirmed compliance fixes and swaps"
        }

        with open(out_json, "w") as f:
            json.dump(manifest, f, indent=2)

        return out_csv, out_json

    
    # Summary in plain language
    # -------------------------
    def summary_report(self) -> str:
        return (
            f"Compliance Summary:\n"
            f"- Swaps suggested: {len(self.suggested_swaps)}\n"
            f"- Swaps applied (confirmed): {self.summary_stats['swaps_applied']}\n"
            f"- Direct fixes: {self.summary_stats['direct_fixes']}\n"
            f"- Meal breaks inserted: {self.summary_stats['meal_breaks_inserted']}\n"
            f"- Penalty rates adjusted: {self.summary_stats['penalty_rates_adjusted']}\n"
        )
