# Constraint Solver Agent (Safe & Context-Aligned)
from ortools.sat.python import cp_model
import pandas as pd

class ConstraintSolverAgent:
    def run(self, context):
        # Pull everything from context
        availability = context["availability_tidy"].copy()
        staff_estimate = context["staff_estimate"]
        MIN_SHIFT_HOURS = context["MIN_SHIFT_HOURS"]
        MIN_REST_HOURS = context["MIN_REST_HOURS"]
        MAX_CONSEC_DAYS = context["MAX_CONSEC_DAYS"]

        # Step 1: Ensure day ordering safely
        # Factorize ensures every label gets a numeric index even if parsing fails
        availability["day_index"] = pd.factorize(availability["day_label"])[0]

        # Step 2: Build CP-SAT model
        model = cp_model.CpModel()
        assignments = {}

        for idx, row in availability.iterrows():
            var = model.NewBoolVar(
                f"assign_{row['employee_id']}_{row['day_label']}_{row['shift_code']}"
            )
            assignments[(row['employee_id'], row['day_label'], row['shift_code'])] = var

        # Constraint 1: Minimum shift length
        for idx, row in availability.iterrows():
            if pd.notnull(row["start_time"]) and pd.notnull(row["end_time"]):
                duration = row["end_time"] - row["start_time"]
                if duration < MIN_SHIFT_HOURS:
                    model.Add(assignments[(row['employee_id'], row['day_label'], row['shift_code'])] == 0)

        # Constraint 2: One shift per employee per day
        for (emp_id, day), group in availability.groupby(["employee_id", "day_label"]):
            vars = [assignments[(emp_id, day, sc)] for sc in group["shift_code"]]
            model.Add(sum(vars) <= 1)

        # Constraint 3: Minimum rest between shifts
        for emp_id, group in availability.groupby("employee_id"):
            group_sorted = group.sort_values("day_index")
            prev_end = None
            prev_var = None
            for _, row in group_sorted.iterrows():
                var = assignments[(row['employee_id'], row['day_label'], row['shift_code'])]
                if prev_end is not None and row["start_time"] is not None:
                    rest = row["start_time"] - prev_end
                    if rest < MIN_REST_HOURS:
                        model.Add(var + prev_var <= 1)
                prev_end = row["end_time"]
                prev_var = var

        # Constraint 4: Maximum consecutive working days
        for emp_id, group in availability.groupby("employee_id"):
            days = sorted(group["day_index"].unique())
            for i in range(len(days) - MAX_CONSEC_DAYS):
                window = days[i:i+MAX_CONSEC_DAYS+1]
                vars = []
                for d in window:
                    for _, row in group[group["day_index"] == d].iterrows():
                        vars.append(assignments[(row['employee_id'], row['day_label'], row['shift_code'])])
                if vars:
                    model.Add(sum(vars) <= MAX_CONSEC_DAYS)

        # Constraint 5: Staff demand coverage (aligned with CSV columns)
        for _, demand in staff_estimate.iterrows():
            required_counts = {
                "Kitchen": int(demand["kitchen_staff"]),
                "Counter": int(demand["counter_staff"]),
                "Multi-Station McCafe": int(demand["mccafe_staff"]),
                "Dessert Station": int(demand["dessert_station_staff"]),
                "Offline Dessert Station": int(demand["offline_dessert_station_staff"])
            }

            for station, required in required_counts.items():
                if required > 0:
                    vars = []
                    for _, row in availability[availability["station"] == station].iterrows():
                        vars.append(assignments[(row['employee_id'], row['day_label'], row['shift_code'])])
                    if vars:
                        model.Add(sum(vars) >= required)

        # Objective: maximize total assignments
        model.Maximize(sum(assignments.values()))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30
        status = solver.Solve(model)

        roster_rows = []
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            for (emp_id, day, sc), var in assignments.items():
                if solver.Value(var) == 1:
                    subset = availability[(availability["employee_id"] == emp_id) &
                                          (availability["day_label"] == day) &
                                          (availability["shift_code"] == sc)]
                    if not subset.empty:
                        match = subset.iloc[0]
                        roster_rows.append({
                            "employee_id": emp_id,
                            "employee_name": match["employee_name"],
                            "day_label": day,
                            "shift_code": sc,
                            "station": match["station"],
                            "start_time": match["start_time"],
                            "end_time": match["end_time"]
                        })

        roster_df = pd.DataFrame(roster_rows)
        context["draft_roster_df"] = roster_df
        return context

