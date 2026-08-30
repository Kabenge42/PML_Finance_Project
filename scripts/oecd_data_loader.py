import requests
import pandas as pd
from io import StringIO

# Define API query URL (CSV with labels format)
url = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE4,2.0/A.SRB+KAZ+IDN+BRA+IND+HKG+CHN+SAU+ZAF+SGP+ROU+MLT+GEO+BGR+EU+AUS+AUT+BEL+CAN+CHL+COL+CRI+CZE+DNK+EST+FIN+FRA+DEU+GRC+HUN+ISL+IRL+ISR+ITA+JPN+KOR+LVA+LTU+LUX+MEX+NLD+NZL+NOR+POL+PRT+SVK+SVN+ESP+SWE+CHE+TUR+GBR+USA...EXC_A.......?startPeriod=2021&format=csvfile"

# Fetch data
response = requests.get(url)

# Load into pandas DataFrame
df = pd.read_csv(StringIO(response.text))

# Display first few rows
print(df.head())