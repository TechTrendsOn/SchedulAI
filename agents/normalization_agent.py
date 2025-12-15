# Normalization_agent.py
import pandas as pd

class NormalizationAgent:
    def run(self, context: dict) -> dict:
        employees_df = context["employees_df"]
        basic_params = context["basic_params"]

        # Step 1: Fix headers and drop intro rows
        employees_df.columns = employees_df.iloc[3]   # 4th row as headers
        employees_df = employees_df[4:].reset_index(drop=True)

        # Step 2: Melt wide availability into tidy rows
        meta_cols = ["ID", "Employee Name", "Type", "Station"]
        date_cols = [c for c in employees_df.columns 
                     if c not in meta_cols and str(c).strip() != ""]
        tidy_rows = []
        for _, row in employees_df.iterrows():
            for dcol in date_cols:
                shift_code = str(row[dcol]).strip()
                tidy_rows.append({
                    "employee_id": row["ID"],
                    "employee_name": row["Employee Name"],
                    "employment_type": row["Type"],
                    "station": row["Station"],
                    "day_label": dcol,
                    "shift_code": shift_code
                })
        availability_tidy = pd.DataFrame(tidy_rows)

        # Step 3: Map shift codes to start/end times
        SHIFT_MAP = {
            "S": ("06:30", "15:00"),
            "1F": ("06:30", "15:30"),
            "2F": ("14:00", "23:00"),
            "3F": ("08:00", "20:00"),
            "SC": ("11:00", "20:00"),
            "/": (None, None),
            "NA": (None, None)
        }

        def parse_time(tstr):
            if tstr is None or pd.isna(tstr):
                return None
            h, m = map(int, tstr.split(":"))
            return h + m/60.0

        availability_tidy["start_time"] = availability_tidy["shift_code"].map(
            lambda s: parse_time(SHIFT_MAP.get(s, (None, None))[0])
        )
        availability_tidy["end_time"] = availability_tidy["shift_code"].map(
            lambda s: parse_time(SHIFT_MAP.get(s, (None, None))[1])
        )

        # Ensure numeric
        availability_tidy["start_time"] = pd.to_numeric(availability_tidy["start_time"], errors="coerce")
        availability_tidy["end_time"]   = pd.to_numeric(availability_tidy["end_time"], errors="coerce")

        # Step 4: Extract canonical parameters
        def get_param(name):
            row = basic_params[basic_params.iloc[:,0].astype(str).str.contains(name, case=False, na=False)]
            if row.empty:
                return None
            return row.iloc[0,1]

        MIN_SHIFT_HOURS = float(get_param("Minimum Hours Per Shift") or 3)
        MIN_REST_HOURS  = float(get_param("Minimum Hours Between Shifts") or 10)
        MAX_CONSEC_DAYS = int(get_param("Maximum Consecutive Working Days") or 6)

        # Step 5: Update context
        context.update({
            "availability_tidy": availability_tidy,
            "MIN_SHIFT_HOURS": MIN_SHIFT_HOURS,
            "MIN_REST_HOURS": MIN_REST_HOURS,
            "MAX_CONSEC_DAYS": MAX_CONSEC_DAYS
        })
        return context

