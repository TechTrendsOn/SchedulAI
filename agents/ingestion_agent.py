# Ingestion_agent.py
import pandas as pd

def clean_table(df, min_cols=2):
    """
    Cleans a dataframe by:
    - Dropping rows that are mostly NaN or too short
    - Resetting headers if the first row looks like column names
    """
    df = df.dropna(how="all")
    df = df[df.count(axis=1) >= min_cols]
    df = df.reset_index(drop=True)
    return df

class IngestionAgent:
    def run(self, context: dict) -> dict:
        # Load parameters (multi-sheet)
        params_xls = pd.ExcelFile("data/australian_restaurant_rostering_parameters.xlsx")
        params_sheets = {sheet: clean_table(pd.read_excel(params_xls, sheet_name=sheet)) 
                         for sheet in params_xls.sheet_names}
        context["basic_params"] = params_sheets.get("Basic Parameters")
        context["service_periods"] = params_sheets.get("Service Periods")
        context["compliance_notes"] = params_sheets.get("Compliance Notes")

        # Load availability
        availability_xls = pd.ExcelFile("data/employee_availability_2weeks.xlsx")
        employees_df = clean_table(pd.read_excel(availability_xls, sheet_name=availability_xls.sheet_names[0]))
        context["employees_df"] = employees_df
        context["availability_sheets"] = {
            sheet: clean_table(pd.read_excel(availability_xls, sheet_name=sheet))
            for sheet in availability_xls.sheet_names
        }

        # Load fixed hours template
        fixed_hours_xls = pd.ExcelFile("data/fixed_hours_template_2columns.xlsx")
        fixed_hours_sheets = {sheet: clean_table(pd.read_excel(fixed_hours_xls, sheet_name=sheet)) 
                              for sheet in fixed_hours_xls.sheet_names}
        context["fixed_hours"] = fixed_hours_sheets.get("Fixed Hours Template - Table 1")

        # Load management roster codes
        mgmt_xls = pd.ExcelFile("data/management_roster_simplified.xlsx")
        mgmt_sheets = {sheet: clean_table(pd.read_excel(mgmt_xls, sheet_name=sheet)) 
                       for sheet in mgmt_xls.sheet_names}
        context["shift_codes"] = mgmt_sheets.get("Shift Codes")
        context["mgmt_roster"] = mgmt_sheets.get("Monthly Roster")

        # Load store configs
        store_xls = pd.ExcelFile("data/store_configurations.xlsx")
        store_sheets = {sheet: clean_table(pd.read_excel(store_xls, sheet_name=sheet)) 
                        for sheet in store_xls.sheet_names}
        context["store_configs"] = store_sheets[store_xls.sheet_names[0]]

        # Load staff demand estimates
        context["staff_estimate"] = pd.read_csv("data/store_structure_staff_estimate.csv")

        return context

