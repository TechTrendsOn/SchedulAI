# Data Ingestion Agent with Cleaning
import pandas as pd

def clean_table(df, min_cols=2):
    """
    Cleans a dataframe by:
    - Dropping rows that are mostly NaN or too short
    - Resetting headers if the first row looks like column names
    """
    # Drop empty rows
    df = df.dropna(how="all")
    # Keep only rows with enough non-empty cells
    df = df[df.count(axis=1) >= min_cols]
    # Reset index
    df = df.reset_index(drop=True)
    return df

# Load parameters (multi-sheet)
params_xls = pd.ExcelFile("australian_restaurant_rostering_parameters.xlsx")
params_sheets = {sheet: clean_table(pd.read_excel(params_xls, sheet_name=sheet)) 
                 for sheet in params_xls.sheet_names}

basic_params = params_sheets.get("Basic Parameters")
service_periods = params_sheets.get("Service Periods")
compliance_notes = params_sheets.get("Compliance Notes")

# Load availability (first sheet, but keep sheet_names for reference)
availability_xls = pd.ExcelFile("employee_availability_2weeks.xlsx")
employees_df = pd.read_excel(availability_xls, sheet_name=availability_xls.sheet_names[0])
employees_df = clean_table(employees_df)
availability_sheets = {sheet: clean_table(pd.read_excel(availability_xls, sheet_name=sheet)) 
                       for sheet in availability_xls.sheet_names}

# Load fixed hours template (multiple sheets)
fixed_hours_xls = pd.ExcelFile("fixed_hours_template_2columns.xlsx")
fixed_hours_sheets = {sheet: clean_table(pd.read_excel(fixed_hours_xls, sheet_name=sheet)) 
                      for sheet in fixed_hours_xls.sheet_names}
fixed_hours = fixed_hours_sheets.get("Fixed Hours Template - Table 1")

# Load management roster codes (multiple sheets)
mgmt_xls = pd.ExcelFile("management_roster_simplified.xlsx")
mgmt_sheets = {sheet: clean_table(pd.read_excel(mgmt_xls, sheet_name=sheet)) 
               for sheet in mgmt_xls.sheet_names}
shift_codes = mgmt_sheets.get("Shift Codes")
mgmt_roster = mgmt_sheets.get("Monthly Roster")

# Load store configs (first sheet only, but keep dictionary for extensibility)
store_xls = pd.ExcelFile("store_configurations.xlsx")
store_sheets = {sheet: clean_table(pd.read_excel(store_xls, sheet_name=sheet)) 
                for sheet in store_xls.sheet_names}
store_configs = store_sheets[store_xls.sheet_names[0]]

# Load staff demand estimates (CSV)
staff_estimate = pd.read_csv("store_structure_staff_estimate.csv")

# Quick previews
print("Basic params:")
display(basic_params.head())
print("Service periods:")
display(service_periods.head())
print("Compliance notes:")
display(compliance_notes.head())
print("Employees availability:")
display(employees_df.head())
print("Fixed hours sheets:")
for name, df in fixed_hours_sheets.items():
    print(f"Sheet: {name}")
    display(df.head())
print("Management sheets:")
for name, df in mgmt_sheets.items():
    print(f"Sheet: {name}")
    display(df.head())
print("Store configs:")
display(store_configs.head())
print("Staff estimate:")
display(staff_estimate.head())

# Load parameters (multi-sheet)
params_xls = pd.ExcelFile("australian_restaurant_rostering_parameters.xlsx")
basic_params = pd.read_excel(params_xls, sheet_name="Basic Parameters")
service_periods = pd.read_excel(params_xls, sheet_name="Service Periods")
compliance_notes = pd.read_excel(params_xls, sheet_name="Compliance Notes")

# Load availability
availability_xls = pd.ExcelFile("employee_availability_2weeks.xlsx")
# The first sheet usually hosts the table; if sheet name is different, print xls.sheet_names
employees_df = pd.read_excel(availability_xls, sheet_name=availability_xls.sheet_names[0])

# Load fixed hours template (optional)
fixed_hours_xls = pd.ExcelFile("fixed_hours_template_2columns.xlsx")
fixed_hours = pd.read_excel(fixed_hours_xls, sheet_name=fixed_hours_xls.sheet_names[0])

# Load management roster codes (optional)
mgmt_xls = pd.ExcelFile("management_roster_simplified.xlsx")
shift_codes = pd.read_excel(mgmt_xls, sheet_name=mgmt_xls.sheet_names[0])

# Load store configs
store_xls = pd.ExcelFile("store_configurations.xlsx")
store_configs = pd.read_excel(store_xls, sheet_name=store_xls.sheet_names[0])

# Load staff demand estimates
staff_estimate = pd.read_csv("store_structure_staff_estimate.csv")

# Quick previews
print("Basic params:")
display(basic_params.head())
print("Service periods:")
display(service_periods.head())
print("Compliance notes:")
display(compliance_notes.head())
print("Employees availability:")
display(employees_df.head())
print("Store configs:")
display(store_configs.head())
print("Staff estimate:")
display(staff_estimate)