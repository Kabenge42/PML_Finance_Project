import pandas as pd
df_dw = pd.read_csv('C:/Users/markm/PML_Finance_Project/data/playground/screening_etf.csv', sep=',', quotechar='"')
# Change Type: `Inception Date`
df_dw['Inception Date'] = df_dw['Inception Date'].astype("datetime64[ns]")
# Change Type: `Launch Date`
df_dw['Launch Date'] = df_dw['Launch Date'].astype("datetime64[ns]")
# Change Type: `Last Updated`
df_dw['Last Updated'] = df_dw['Last Updated'].astype("datetime64[ns]")
# Drop Column: `Annualized excess returns`
df_dw = df_dw.drop(columns=['Annualized excess returns'])
# Drop Column: `Annualized returns`
df_dw = df_dw.drop(columns=['Annualized returns'])
# Drop Column: `12B-1`
df_dw = df_dw.drop(columns=['12B-1'])

df_dw.head()