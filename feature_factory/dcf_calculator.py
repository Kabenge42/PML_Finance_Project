# This scripts allow you to upload financial statements in .csv format (e.g., downloaded from Morningstar.com), and automatically estimate the fair price of the company based on a discounted cash flow model. The advantage is that you don't have to manually import all the data into a spreadsheet-based DCF Model, then saving time and reducing the likelihood of making mistakes (which doesn't eliminate completely the possibility of making mistakes). The fair price depend on different variables, such as the WACC you decide (which you can change in the formula). Moreover, the compounded growth of each financial value is calculated based on a weighted average of the last few years, that gives bigger weights to certin year (e.g., in this example it gives smaller weights to the COVID-19 years, because they caused anomalies in the economy.

# This script is by no means a predictor and it isn't a magic formula for financial success, it's up to you to change the weights in the weighted average YOY growth, to estimate the WACC, to interpret the results, and to spot any bug in the code. If you find any bug in the code, please let me know, I'll be happy to fix it and improve the code. Moreover, it's impossible to predict the future, and you should be cautious when interpreting the fair price and any other result calculated by this script.

# This script uses Roche Holding AG (RHHBY) as an example,and you may have to change the attribute names when importing your own three financial statements, because those attributed may be named differently (e.g., Total Revenue could be named Revenue or have other names).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Load financial statements data from CSV files
income_statement = pd.read_csv(
    "RHHBY_Income Statement_NEW_nodatetime.csv", header=0, index_col=0, usecols=[0, 1, 2, 3, 4, 5]
).transpose()
balance_sheet = pd.read_csv(
    "RHHBY_Balance Sheet_NEW_nodatetime.csv", header=0, index_col=0, usecols=[0, 1, 2, 3, 4, 5]
).transpose()
cash_flow_statement = pd.read_csv(
    "RHHBY_Cash Flow_NEW_nodatetime.csv", header=0, index_col=0, usecols=[0, 1, 2, 3, 4, 5]
).transpose()

print(income_statement.head())
print(balance_sheet.head())
print(cash_flow_statement.head())

income_statement.to_csv("RHHBY_Income Statement_transposed.csv", index=True)
balance_sheet.to_csv("RHHBY_Balance Sheet_transposed.csv", index=True)
cash_flow_statement.to_csv("RHHBY_Cash Flow Statement_transposed.csv", index=True)

# Define the discount rate (e.g. the weighted average cost of capital)
discount_rate = 0.2

# Input the number of shares outstanding
shares_outstanding = 6405000000

# Calculate the net income
revenue = income_statement["Total Revenue"].iloc[-1]
cost_of_goods_sold = income_statement["Cost of Revenue"].iloc[-1]
gross_profit = revenue - cost_of_goods_sold

operating_expenses = income_statement["Operating Income/Expenses"].iloc[-1]
operating_income = gross_profit - operating_expenses

other_income = income_statement["Other Income/Expense, Operating"].iloc[-1]
interest_expense = income_statement["Net Interest Income/Expense"].iloc[-1]
income_before_tax = operating_income + other_income - interest_expense

income_tax = income_statement["Provision for Income Tax"].iloc[-1]
net_income = income_before_tax - income_tax

print(f"Net Income is: ${net_income:.2f}")

# calculate the net income per each year
revenue_all = income_statement["Total Revenue"]
cost_of_goods_sold_all = income_statement["Cost of Revenue"]
gross_profit_all = revenue_all - cost_of_goods_sold_all

operating_expenses_all = income_statement["Operating Income/Expenses"]
operating_income_all = gross_profit_all - operating_expenses_all

other_income_all = income_statement["Other Income/Expense, Operating"]
interest_expense_all = income_statement["Net Interest Income/Expense"]
income_before_tax_all = operating_income_all + other_income_all - interest_expense_all

income_tax_all = income_statement["Provision for Income Tax"]
net_income_all = income_before_tax_all - income_tax_all

print("Net Income for each year:")
print(net_income_all)


# Perform calculations to derive free cash flow
capital_expenditures = cash_flow_statement[
    "Purchase/Sale and Disposal of Property, Plant and Equipment, Net"
].iloc[-1]

# Calculate change in working capital
current_assets = balance_sheet["Total Current Assets"].iloc[-1]
current_liabilities = balance_sheet["Total Current Liabilities"].iloc[-1]
working_capital = current_assets - current_liabilities
print("The present working capital is: ")
print(working_capital)

# Initialize an empty list to store historical working capital values
historical_working_capital = []

# Iterate over the rows of the balance_sheet DataFrame
for index, row in balance_sheet.iterrows():
    current_assets = row["Total Current Assets"]
    current_liabilities = row["Total Current Liabilities"]
    working_capital = current_assets - current_liabilities
    historical_working_capital.append(working_capital)

# Convert the list to a NumPy array for further processing if needed
historical_working_capital = np.array(historical_working_capital)


current_assets_prev = balance_sheet["Total Current Assets"].iloc[-2]

current_liabilities_prev = balance_sheet["Total Current Liabilities"].iloc[-2]
change_in_working_capital = (
    current_assets - current_assets_prev + current_liabilities_prev - current_liabilities
)

current_assets_prev = balance_sheet["Total Current Assets"].iloc[-2]

depreciation_amortization = cash_flow_statement[
    "Depreciation, Amortization and Depletion, Non-Cash Adjustment"
].iloc[-1]

free_cash_flow = (
    net_income + depreciation_amortization - capital_expenditures - change_in_working_capital
)


# -----------------------------------------------------------------------------------------------------------
# WACC #let's not use it for the moment

# Get the required values
equity = balance_sheet["Total Equity"]
debt = balance_sheet["Total Liabilities"]
assets = balance_sheet["Total Assets"]
interest_expense = balance_sheet["Interest Payable, Current"]

risk_free_rate = 0.0128
market_risk_premium = 0.0580
levered_beta = 0.78
size_premium = 0.017

# Calculate #these numbers are random and should be calculated properly
cost_of_equity = risk_free_rate + (market_risk_premium * levered_beta) + size_premium
cost_of_debt = interest_expense / debt
tax_rate = 0.085

debt_to_equity_ratio = debt / equity
equity_weight = equity / (equity + debt)
debt_weight = debt / (equity + debt)
after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)

WACC = equity_weight * cost_of_equity + debt_weight * after_tax_cost_of_debt
print("The WACC is: ")
print(WACC)
# ------------------------------------------------------------------------------------------------------------------------------
# """
# Average growth method
# Historic Growth
# Define the number of years for which we want to calculate YoY growth
num_years = 5
# Define the number of years for which we'll be projecting cash flows
projection_years = 5

# Define the weights for each year
# weights_a = np.array([0.20, 0.20, 0.25, 0.15, 0.15])
weights_a = np.array([0.30, 0.30, 0.20, 0.20])

# Total revenue
# Calculate the YoY growth rate for each year
revenue_history = income_statement["Total Revenue"][-num_years:]
revenue_yoy_growth = [
    (revenue_history.iloc[i] - revenue_history.iloc[i - 1]) / revenue_history.iloc[i - 1]
    for i in range(1, num_years)
]

# Calculate the weighted average YoY growth rate
revenue_weighted_average_yoy_growth = sum(
    [revenue_yoy_growth[i] * weights_a[i] for i in range(num_years - 1)]
) / sum(weights_a)
print(
    f"The weighted average YoY growth rate for the revenue over the past {num_years} years is: {revenue_weighted_average_yoy_growth:.2%}"
)

# Predict future growth based on the weighted average YoY growth rate
projected_revenue = [
    revenue_history.iloc[-1] * (1 + revenue_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]

# Cost of revenue
# Calculate the YoY growth rate for each year
cost_of_revenue_history = income_statement["Cost of Revenue"][-num_years:]
cost_of_revenue_yoy_growth = [
    (cost_of_revenue_history.iloc[i] - cost_of_revenue_history.iloc[i - 1])
    / cost_of_revenue_history.iloc[i - 1]
    for i in range(1, num_years)
]
# cost_of_revenue_yoy_growth.append(cost_of_revenue_yoy_growth[-1])  # Add the growth rate for the projected year
print(
    f"The YoY growth rate for the cost of revenue over the past {num_years} years is: {cost_of_revenue_yoy_growth:}"
)

# Calculate the weighted average YoY growth rate
cost_of_revenue_weighted_average_yoy_growth = sum(
    [cost_of_revenue_yoy_growth[i] * weights_a[i] for i in range(num_years - 1)]
) / sum(weights_a)
print(
    f"The weighted average YoY growth rate for the revenue over the past {num_years} years is: {cost_of_revenue_weighted_average_yoy_growth:.2%}"
)

# Growth Projection
projected_cost_of_revenue = [
    cost_of_goods_sold * (1 + cost_of_revenue_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]

# Operating expenses
# Calculate the YoY growth rate for each year
operating_expenses_history = income_statement["Operating Income/Expenses"][-num_years:]
operating_expenses_yoy_growth = [
    (operating_expenses_history.iloc[i] - operating_expenses_history.iloc[i - 1])
    / operating_expenses_history.iloc[i - 1]
    for i in range(1, num_years)
]
# operating_expenses_yoy_growth.append(operating_expenses_yoy_growth[-1])  # Add the growth rate for the projected year
print(
    f"The YoY growth rate for the operating expenses over the past {num_years} years is: {operating_expenses_yoy_growth:}"
)

# Calculate the weighted average YoY growth rate
operating_expenses_weighted_average_yoy_growth = sum(
    [operating_expenses_yoy_growth[i] * weights_a[i] for i in range(num_years - 1)]
) / sum(weights_a)
print(
    f"The weighted average YoY growth rate for the operating expenses over the past {num_years} years is: {operating_expenses_weighted_average_yoy_growth:.2%}"
)

# Growth Projection
projected_operating_expenses = [
    operating_expenses * (1 + operating_expenses_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]

# Other Income
# Calculate the YoY growth rate for each year
other_income_history = income_statement["Other Income/Expense, Operating"][-num_years:]
other_income_yoy_growth = [
    (other_income_history.iloc[i] - other_income_history.iloc[i - 1])
    / other_income_history.iloc[i - 1]
    for i in range(1, num_years)
]
# other_income_yoy_growth.append(other_income_yoy_growth[-1])  # Add the growth rate for the projected year
print(
    f"The YoY growth rate for the other income over the past {num_years} years is: {other_income_yoy_growth:}"
)

# Calculate the weighted average YoY growth rate
other_income_weighted_average_yoy_growth = sum(
    [other_income_yoy_growth[i] * weights_a[i] for i in range(num_years - 1)]
) / sum(weights_a)
print(
    f"The weighted average YoY growth rate for the other income over the past {num_years} years is: {other_income_weighted_average_yoy_growth:.2%}"
)

# Growth Projection
projected_other_income = [
    other_income * (1 + other_income_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]

# Interest Expense
# Calculate the YoY growth rate for each year
interest_expense_history = income_statement["Net Interest Income/Expense"][-num_years:]
interest_expense_yoy_growth = [
    (interest_expense_history.iloc[i] - interest_expense_history.iloc[i - 1])
    / interest_expense_history.iloc[i - 1]
    for i in range(1, num_years)
]
# interest_expense_yoy_growth.append(interest_expense_yoy_growth[-1])  # Add the growth rate for the projected year
print(
    f"The YoY growth rate for interest expense over the past {num_years} years is: {interest_expense_yoy_growth:}"
)

# Calculate the weighted average YoY growth rate
interest_expense_weighted_average_yoy_growth = sum(
    [interest_expense_yoy_growth[i] * weights_a[i] for i in range(num_years - 1)]
) / sum(weights_a)
print(
    f"The weighted average YoY growth rate for interest expense over the past {num_years} years is: {interest_expense_weighted_average_yoy_growth:.2%}"
)

# Growth Projection
projected_interest_expense = [
    interest_expense * (1 + interest_expense_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]

# Income Tax
# Calculate the YoY growth rate for each year
income_tax_history = income_statement["Provision for Income Tax"][-num_years:]
income_tax_yoy_growth = [
    (income_tax_history.iloc[i] - income_tax_history.iloc[i - 1]) / income_tax_history.iloc[i - 1]
    for i in range(1, num_years)
]
# income_tax_yoy_growth.append(income_tax_yoy_growth[-1])  # Add the growth rate for the projected year
print(
    f"The YoY growth rate for income tax over the past {num_years} years is: {income_tax_yoy_growth}"
)

# Calculate the weighted average YoY growth rate
income_tax_average_yoy_growth = sum(
    [income_tax_yoy_growth[i] * weights_a[i] for i in range(num_years - 1)]
) / sum(weights_a)
print(
    f"The weighted average YoY growth rate for income tax over the past {num_years} years is: {income_tax_average_yoy_growth:.2%}"
)

# Growth Projection
projected_income_tax = [
    income_tax * (1 + income_tax_average_yoy_growth) ** (i + 1) for i in range(projection_years)
]


# capex
# Calculate the YoY growth rate for each year
capex_history = cash_flow_statement[
    "Purchase/Sale and Disposal of Property, Plant and Equipment, Net"
][-num_years:]
capex_yoy_growth = [
    (capex_history.iloc[i] - capex_history.iloc[i - 1]) / capex_history.iloc[i - 1]
    for i in range(1, num_years)
]
# capex_yoy_growth.append(capex_yoy_growth[-1])  # Add the growth rate for the projected year
print(
    f"The YoY growth rate for capital expenditures over the past {num_years} years is: {capex_yoy_growth}"
)

# Calculate the weighted average YoY growth rate
capex_weighted_average_yoy_growth = sum(
    [capex_yoy_growth[i] * weights_a[i] for i in range(num_years - 1)]
) / sum(weights_a)
print(
    f"The weighted average YoY growth rate for capital expenditures over the past {num_years} years is: {capex_weighted_average_yoy_growth:.2%}"
)

# Growth Projection
projected_capex = [
    capital_expenditures * (1 + capex_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]


# Current assets
# Current assets
# Calculate the YoY growth rate for each year
current_assets_history = balance_sheet["Total Current Assets"][-num_years:]
current_assets_yoy_growth = [
    (current_assets_history.iloc[i] - current_assets_history.iloc[i - 1])
    / current_assets_history.iloc[i - 1]
    for i in range(1, num_years)
]
# current_assets_yoy_growth.append(current_assets_yoy_growth[-1])  # Add the growth rate for the projected year
print(
    f"The YoY growth rate for current assets over the past {num_years} years is: {current_assets_yoy_growth}"
)

# Calculate the weighted average YoY growth rate
weights_a = [1 / num_years] * num_years  # Equal weights_a for each year
current_assets_weighted_average_yoy_growth = sum(
    [current_assets_yoy_growth[i] * weights_a[i] for i in range(num_years - 1)]
) / sum(weights_a)
print(
    f"The weighted average YoY growth rate for current assets over the past {num_years} years is: {current_assets_weighted_average_yoy_growth:.2%}"
)

# Growth Projection
projected_current_assets = [
    current_assets * (1 + current_assets_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]

# Current Liabilities
# Calculate the YoY growth rate for each year
current_liabilities_history = balance_sheet["Total Current Liabilities"][-num_years:]
current_liabilities_yoy_growth = [
    (current_liabilities_history.iloc[i] - current_liabilities_history.iloc[i - 1])
    / current_liabilities_history.iloc[i - 1]
    for i in range(1, num_years)
]
# current_liabilities_yoy_growth.append(current_liabilities_yoy_growth[-1])  # Add the growth rate for the projected year
print(
    f"The YoY growth rate for current liabilities over the past {num_years} years is: {current_liabilities_yoy_growth}"
)

# Calculate the weighted average YoY growth rate
current_liabilities_weighted_average_yoy_growth = sum(
    [current_liabilities_yoy_growth[i] * weights_a[i] for i in range(num_years - 1)]
) / sum(weights_a)
print(
    f"The weighted average YoY growth rate for current liabilities over the past {num_years} years is: {current_liabilities_weighted_average_yoy_growth:.2%}"
)

# Growth Projection
projected_current_liabilities = [
    current_liabilities * (1 + current_liabilities_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]

# depreciation_amortization
# Depreciation & Amortization
# Calculate the YoY growth rate for each year
depreciation_history = cash_flow_statement[
    "Depreciation, Amortization and Depletion, Non-Cash Adjustment"
][-num_years:]
depreciation_yoy_growth = [
    (depreciation_history.iloc[i] - depreciation_history.iloc[i - 1])
    / depreciation_history.iloc[i - 1]
    for i in range(1, num_years)
]
# depreciation_yoy_growth.append(depreciation_yoy_growth[-1])  # Add the growth rate for the projected year
print(
    f"The YoY growth rate for Depreciation & Amortization over the past {num_years} years is: {depreciation_yoy_growth}"
)

# Calculate the weighted average YoY growth rate
depreciation_weighted_average_yoy_growth = sum(
    [depreciation_yoy_growth[i] * weights_a[i] for i in range(num_years - 1)]
) / sum(weights_a)
print(
    f"The weighted average YoY growth rate for Depreciation & Amortization over the past {num_years} years is: {depreciation_weighted_average_yoy_growth:.2%}"
)

# Growth Projection
projected_depreciation_amortization = [
    depreciation_amortization * (1 + depreciation_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]

# Change in working capital
# Change in working capital
# Calculate the YoY growth rate for each year
working_capital_history = [
    balance_sheet["Total Current Assets"].iloc[-(i + 1)]
    - balance_sheet["Total Current Liabilities"].iloc[-(i + 1)]
    for i in range(num_years)
]
working_capital_yoy_growth = [
    (working_capital_history[i] - working_capital_history[i - 1]) / working_capital_history[i - 1]
    for i in range(1, num_years)
]
print(
    f"The YoY growth rate for working capital over the past {num_years} years is: {working_capital_yoy_growth}"
)

# Calculate the weighted average YoY growth rate
working_capital_weights_a = [
    weights_a[i] + weights_a[i + 1] for i in range(num_years - 1)
]  # why? Replace when you reduce the weights_a array from 5 to 4 numbers
working_capital_weighted_average_yoy_growth = sum(
    [working_capital_yoy_growth[i] * working_capital_weights_a[i] for i in range(num_years - 1)]
) / sum(working_capital_weights_a)
print(
    f"The weighted average YoY growth rate for working capital over the past {num_years} years is: {working_capital_weighted_average_yoy_growth:.2%}"
)

# Growth Projection
projected_working_capital = [
    working_capital * (1 + working_capital_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]

# net income (trial)
# net_income_yoy_growth = revenue_yoy_growth - cost_of_revenue_yoy_growth - operating_expenses_yoy_growth + other_income_yoy_growth - interest_expense_yoy_growth - income_tax_yoy_growth
# print(net_income_yoy_growth)

# Create arrays for projected values
p_revenues = []
p_cost_of_goods_solds = []
p_gross_profits = []
p_operating_expenseses = []
p_operating_incomes = []
p_other_incomes = []
p_interest_expenses = []
p_income_before_taxes = []
p_income_taxes = []
p_net_incomes = []
p_current_assetses = []
p_current_liabilitiess = []
p_working_capitals = []
p_capexes = []
p_depreciations_amortizations = []

# Calculate projected values for each year and append to arrays
for i in range(1, projection_years + 1):
    # p_revenue = revenue * (1 + revenue_average_yoy_growth) ** i
    p_revenue = revenue * (1 + revenue_weighted_average_yoy_growth) ** i
    p_cost_of_goods_sold = (
        cost_of_goods_sold * (1 + cost_of_revenue_weighted_average_yoy_growth) ** i
    )
    p_gross_profit = p_revenue - p_cost_of_goods_sold
    p_operating_expenses = (
        operating_expenses * (1 + operating_expenses_weighted_average_yoy_growth) ** i
    )
    p_operating_income = p_gross_profit - p_operating_expenses
    p_other_income = other_income * (1 + other_income_weighted_average_yoy_growth) ** i
    p_interest_expense = interest_expense * (1 + interest_expense_weighted_average_yoy_growth) ** i
    p_income_before_tax = p_operating_income + p_other_income - p_interest_expense
    p_income_tax = income_tax * (1 + income_tax_average_yoy_growth) ** i
    # p_net_income = p_income_before_tax - p_income_tax
    p_current_assets = current_assets * (1 + current_assets_weighted_average_yoy_growth) ** i
    p_current_liabilities = (
        current_liabilities * (1 + current_liabilities_weighted_average_yoy_growth) ** i
    )
    p_working_capital = p_current_assets - p_current_liabilities
    p_capex = capital_expenditures * (1 + capex_weighted_average_yoy_growth) ** i
    p_depreciation_amortization = (
        depreciation_amortization * (depreciation_weighted_average_yoy_growth) ** i
    )

    # Append values to arrays
    p_revenues.append(p_revenue)
    p_cost_of_goods_solds.append(p_cost_of_goods_sold)
    p_gross_profits.append(p_gross_profit)
    p_operating_expenseses.append(p_operating_expenses)
    p_operating_incomes.append(p_operating_income)
    p_other_incomes.append(p_other_income)
    p_interest_expenses.append(p_interest_expense)
    p_income_before_taxes.append(p_income_before_tax)
    p_income_taxes.append(p_income_tax)
    # p_net_incomes.append(p_net_income)
    p_current_assetses.append(p_current_assets)
    p_current_liabilitiess.append(p_current_liabilities)
    p_working_capitals.append(p_working_capital)
    p_capexes.append(p_capex)
    p_depreciations_amortizations.append(p_depreciation_amortization)

print("The projected revenue is: ")
print(p_revenues)
print("The projected cost of revenue is: ")
print(p_cost_of_goods_solds)
print("The projected gross profit is: ")
print(p_gross_profits)
print("The projected operating expenses is: ")
print(p_operating_expenseses)
print("The projected operating income is: ")
print(p_operating_incomes)
print("The projected other income is: ")
print(p_other_incomes)
print("The projected interest expenses is: ")
print(p_interest_expenses)
print("The projected income before tax is: ")
print(p_income_before_taxes)
print("The projected income tax is: ")
print(p_income_taxes)
# print("The projected operating net income is: ")
# print(p_net_incomes)
print("The projected current assets is: ")
print(p_current_assetses)
print("The projected current liabilities is: ")
print(p_current_liabilitiess)
print("The projected working capital is: ")
print(p_working_capitals)
print("The projected capex is: ")
print(p_capex)
print("The projected deprecation and amortization is: ")
print(p_depreciation_amortization)

# Calculate net income for each year in the DataFrame
net_income_array = []
cash_inflows_array = []
cash_outflows_array = []
for i, row in income_statement.iterrows():
    revenue = row["Total Revenue"]
    cost_of_goods_sold = row["Cost of Revenue"]
    operating_expenses = row["Operating Income/Expenses"]
    other_income = row["Other Income/Expense, Operating"]
    net_interest_expense_income = row["Net Interest Income/Expense"]
    interest_expense = row["Interest Expense Net of Capitalized Interest"]
    interest_income = row["Interest Income"]
    income_tax = row["Provision for Income Tax"]

    cash_inflows = revenue + other_income + interest_income
    cash_outflows = cost_of_goods_sold + operating_expenses + interest_expense + income_tax

    cash_inflows_array.append(cash_inflows)
    cash_outflows_array.append(cash_outflows)
    # this is all "+" because expenses are already reported as negative values in the csv dataframe
    gross_profit = revenue + cost_of_goods_sold
    operating_income = gross_profit + operating_expenses
    income_before_tax = operating_income + other_income + net_interest_expense_income
    hh_net_income = income_before_tax + income_tax

    net_income_array.append(hh_net_income)

# Print net income array
print("Net Income for Each Year: ")
print(np.array(net_income_array))

# Calculate YoY growth for each year in the net income array #try to avoid the last element
yoy_growth_array = []
# for i in range(1, len(net_income_array)): #this thakes the lenght of the dataset whereas the code below considers only the manual imput of the years we want to analyze
for i in range(
    1, num_years
):  # here the rage could simply be num_years, so there's no need to change the data set or remove a column from the csv file, because you can select the num of years like done above.
    yoy_growth = (net_income_array[i] / net_income_array[i - 1]) - 1
    yoy_growth_array.append(yoy_growth)

# Define array of weights
# weights = [0.28, 0.29, 0.18, 0.29, 0.06]
weights = [0.30, 0.30, 0.20, 0.20]

# Calculate weighted average YoY growth
net_income_weighted_average_yoy_growth = sum([a * b for a, b in zip(yoy_growth_array, weights)])
yoy_projected_net_income = []
for i in range(1, 6):
    pp_net_income = hh_net_income * (1 + net_income_weighted_average_yoy_growth)
    yoy_projected_net_income.append(pp_net_income)
    last_net_income = pp_net_income

# Print results
print("Net Income for Each Year: ")
print(np.array(net_income_array))
print("Net Income YoY Growth for Each Year: ")
print(np.array(yoy_growth_array))
# print(f"YoY Average Growth: {net_income_yoy_average_growth}")
print("Net Income Weighted Average YoY Growth: ", net_income_weighted_average_yoy_growth)
print("Projected Net Income for Next 5 Years: ")
print(np.array(yoy_projected_net_income))

# ------------------------------------------------------------------------------------------------------------------------------

# Calculations
# Recalculate net income based on projected revenue
projected_revenue = projected_revenue[0]
projected_cost_of_goods_sold = projected_cost_of_revenue[0]
# cost_of_goods_sold = income_statement['Cost of Revenue'].iloc[-1]
projected_gross_profit = projected_revenue - projected_cost_of_goods_sold

# operating_expenses = income_statement['Operating Income/Expenses'].iloc[-1]
projected_operating_expenses = projected_operating_expenses[0]
projected_operating_income = projected_gross_profit - projected_operating_expenses

# other_income = income_statement['Other Income/Expense, Operating'].iloc[-1]
projected_other_income = projected_other_income[0]
# interest_expense = income_statement['Net Interest Income/Expense'].iloc[-1]
projected_interest_expense = projected_interest_expense[0]
projected_income_before_tax = (
    projected_operating_income + projected_other_income - projected_interest_expense
)

# income_tax = income_statement['Provision for Income Tax'].iloc[-1]
projected_income_tax = projected_income_tax[0]
projected_net_income = projected_income_before_tax - projected_income_tax

print(f"Net Income based on projected revenue is: ${net_income:.2f}")

# Recalculate free cash flow based on projected revenue and net income
# capital_expenditures = cash_flow_statement['Purchase/Sale and Disposal of Property, Plant and Equipment, Net'].iloc[-1]
projected_capital_expenditures = projected_capex[0]

# current_assets = balance_sheet['Total Current Assets'].iloc[-1]
projected_current_assets = projected_current_assets[0]
# projected_current_assets_prev = projected_current_assets[-1] #is it [-1] or [+1]?

# current_liabilities = balance_sheet['Total Current Liabilities'].iloc[-1]
projected_current_liabilities = projected_current_liabilities[0]
# projected_current_liabilities_prev = projected_current_liabilities[-1] #is it [-1] or [+1]?

projected_working_capital = (
    projected_current_assets - projected_current_liabilities
)  # This is projected_working_capital

current_assets_prev = balance_sheet["Total Current Assets"].iloc[-2]

current_liabilities_prev = balance_sheet["Total Current Liabilities"].iloc[-2]
change_in_working_capital = (
    current_assets - current_assets_prev + current_liabilities_prev - current_liabilities
)
# projected_change_in_working_capital = projected_current_assets - projected_current_assets_prev + projected_current_liabilities - projected_current_liabilities_prev

# Projected change in working capital
projected_working_capital = projected_current_assets - projected_current_liabilities
projected_change_in_working_capital = (
    projected_working_capital - working_capital_history[-1]
)  # verify it is correct
print(
    f"The projected change in working capital in {projection_years} years is: {projected_change_in_working_capital:.2f}"
)

projected_depreciation_amortization = projected_depreciation_amortization[0]

projected_free_cash_flow = (
    projected_net_income
    + projected_depreciation_amortization
    - projected_capital_expenditures
    - projected_change_in_working_capital
)

print(f"Free Cash Flow based on projected revenue is: ${free_cash_flow:.2f}")

# Calculate projected change in working capital using projected values arrays
projected_change_in_working_capital = []
initial_working_capital = historical_working_capital[
    -1
]  # Use last historical value as initial value
for i in range(1, projection_years + 1):
    if i == 1:
        projected_change = p_working_capitals[i - 1] - initial_working_capital
    else:
        projected_change = p_working_capitals[i - 1] - p_working_capitals[i - 2]
    projected_change_in_working_capital.append(projected_change)
print("Length of projected_change_in_working_capital:", len(projected_change_in_working_capital))

# Create a dictionary to store the variables
my_dict = {
    "yoy_projected_net_income": yoy_projected_net_income,
    "p_capexes": p_capexes,
    "p_depreciations_amortizations": p_depreciations_amortizations,
    "projected_change_in_working_capital": projected_change_in_working_capital,
}

# Loop through the dictionary and print the lengths of the lists
for key, value in my_dict.items():
    print("Length of", key, ":", len(value))

# fair value & sensitivity analysis without rendundancies
# Calculate future cash flows using projected values arrays
future_cash_flows = []
for i in range(projection_years - 1):
    future_cash_flow = (
        yoy_projected_net_income[i - 1]
        + p_capexes[i - 1]
        + p_depreciations_amortizations[i - 1]
        - projected_change_in_working_capital[i + 1]
    )
    future_cash_flows.append(future_cash_flow)

# Calculate the present value of future cash flows using a discount rate
discount_rate = 0.20  # Assume a discount rate of 20%
discounted_cash_flows = [
    future_cash_flow / (1 + discount_rate) ** (i + 1)
    for i, future_cash_flow in enumerate(future_cash_flows)
]

# Calculate the fair value of the company
fair_value = sum(discounted_cash_flows)

# Calculate the stock price
stock_price = fair_value / shares_outstanding

# Print the number of years used in the calculations
# num_years = projection_years - 1
num_years = len(future_cash_flows)
print(f"Years used in calculations: {num_years}")

print(f"Fair value of the company: ${fair_value:.2f}")
print(f"Stock price: ${stock_price:.2f}")

# Sensitivity Matrix
# Define range of discount rates
discount_rates = np.arange(0.05, 0.31, 0.01)

# Calculate the fair price of the stock for each discount rate
fair_prices = []
for rate in discount_rates:
    present_value = 0
    for i in range(len(future_cash_flows)):
        present_value += future_cash_flows[i] / ((1 + rate) ** (i + 1))
    fair_price = present_value / shares_outstanding
    fair_prices.append(fair_price)

# Print fair prices for each discount rate
print("Fair Prices for Different Discount Rates:")
for i in range(len(discount_rates)):
    print("Discount Rate: {:.2%}, Fair Price: ${:,.2f}".format(discount_rates[i], fair_prices[i]))

# Plotting
# Historic values plots
years_back = range(1, 6)
years_back_bs = range(1, 6)

# Reset the index of the DataFrame
# income_statement = income_statement.reset_index()
# Get the x-axis tick labels
# x_tick_labels = income_statement.iloc[0:6, 0].tolist()
x_tick_labels = income_statement.index.tolist()

# Set seaborn style
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(years_back, income_statement["Total Revenue"], label="Total Revenue")
ax.plot(years_back, income_statement["Cost of Revenue"], label="Cost of revenue")
ax.plot(years_back, income_statement["Operating Income/Expenses"], label="Operating expenses")
ax.plot(years_back, income_statement["Other Income/Expense, Operating"], label="Other income")
ax.plot(years_back, income_statement["Net Interest Income/Expense"], label="Interest expense")
ax.plot(years_back, income_statement["Provision for Income Tax"], label="Income tax")
ax.plot(years_back, net_income_array, label="Net Income")
ax.plot(
    years_back,
    cash_flow_statement["Purchase/Sale and Disposal of Property, Plant and Equipment, Net"],
    label="Capital expenditures",
)
ax.plot(
    years_back,
    cash_flow_statement["Depreciation, Amortization and Depletion, Non-Cash Adjustment"],
    label="Depreciation & amortization",
)
ax.plot(years_back_bs, balance_sheet["Total Current Assets"], label="Current assets")
ax.plot(years_back_bs, balance_sheet["Total Current Liabilities"], label="Current liabilities")
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
ax.yaxis.set_major_locator(plt.MaxNLocator(20))
ax.set_xlabel("Year")
ax.set_ylabel("Amount")

# Set the x-axis tick labels
ax.set_xticks(years_back)
ax.set_xticklabels(x_tick_labels)

# Set the position of the title
ax.set_title("Historic Values", y=1)

# Set the position of the legend
legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize="medium")
legend.set_bbox_to_anchor((0.5, 1.15))

# Adjust the layout of the plot to make room for the legend
plt.tight_layout()

# refresh/override projected values
projected_revenue = [
    revenue * (1 + revenue_weighted_average_yoy_growth) ** (i + 1) for i in range(projection_years)
]
# projected_revenue = [revenue * (1 + revenue_average_yoy_growth)**(i+1) for i in range(projection_years)]
projected_cost_of_revenue = [
    cost_of_goods_sold * (1 + cost_of_revenue_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]
projected_operating_expenses = [
    operating_expenses * (1 + operating_expenses_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]
projected_other_income = [
    other_income * (1 + other_income_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]
projected_interest_expense = [
    interest_expense * (1 + interest_expense_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]
projected_income_tax = [
    income_tax * (1 + income_tax_average_yoy_growth) ** (i + 1) for i in range(projection_years)
]
projected_capex = [
    capital_expenditures * (1 + capex_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]
projected_current_assets = [
    current_assets * (1 + current_assets_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]
projected_current_liabilities = [
    current_liabilities * (1 + current_liabilities_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]
projected_depreciation_amortization = [
    depreciation_amortization * (1 + depreciation_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]
projected_working_capital = [
    working_capital * (1 + working_capital_weighted_average_yoy_growth) ** (i + 1)
    for i in range(projection_years)
]

# Plot projected values
# Set seaborn style
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(12, 8))

# Define colors for each line
colors = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
    "#BC5E00",
    "#003C30",
    "#1B2ACC",
    "#FF7F00",
    "#2BDE73",
    "#D11212",
    "#882255",
    "#8B8B8B",
    "#2B2B2B",
    "#FFB5B8",
    "#ADFF2F",
]

ax.set_prop_cycle("color", colors)

# Find the last non-TTM year
last_year = int(income_statement.index[-2])
# x_tick_labels_projected = list(range(last_year + 1, last_year + projection_years + 1))
x_tick_labels_projected = list(range(last_year + 2, last_year + 2 + projection_years))

ax.plot(range(1, projection_years + 1), p_revenues, label="Projected Revenue")
ax.plot(range(1, projection_years + 1), p_cost_of_goods_solds, label="Projected Cost of Goods Sold")
ax.plot(range(1, projection_years + 1), p_gross_profits, label="Projected Gross Profit")
ax.plot(
    range(1, projection_years + 1), p_operating_expenseses, label="Projected Operating Expenses"
)
ax.plot(range(1, projection_years + 1), p_operating_incomes, label="Projected Operating Income")
ax.plot(range(1, projection_years + 1), p_other_incomes, label="Projected Other Income")
ax.plot(range(1, projection_years + 1), p_interest_expenses, label="Projected Interest Expense")
ax.plot(range(1, projection_years + 1), p_income_before_taxes, label="Projected Income Before Tax")
ax.plot(range(1, projection_years + 1), p_income_taxes, label="Projected Income Tax")
# ax.plot(range(1, projection_years + 1), p_net_incomes, label='Projected Net Income')
ax.plot(range(1, projection_years + 1), yoy_projected_net_income, label="Projected Net Income")
ax.plot(range(1, projection_years + 1), p_current_assetses, label="Projected Current Assets")
ax.plot(
    range(1, projection_years + 1), p_current_liabilitiess, label="Projected Current Liabilities"
)
ax.plot(range(1, projection_years + 1), p_working_capitals, label="Projected Working Capital")
ax.plot(range(1, projection_years + 1), p_capexes, label="Projected Capex")
ax.plot(
    range(1, projection_years + 1),
    p_depreciations_amortizations,
    label="Projected Depreciation & Amortization",
)
ax.plot(range(1, projection_years), future_cash_flows, label="Future Cash Flows")
ax.plot(range(1, projection_years), discounted_cash_flows, label="Discounted Cash Flows")
# ax.plot(range(1, projection_years + 1), [fair_value] * 5, label='Fair Value')
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
ax.yaxis.set_major_locator(plt.MaxNLocator(20))
ax.set_xlabel("Year")
ax.set_ylabel("Amount")

# Set the position of the title
ax.set_title("Projected Values", y=1)

# Set the position of the legend
legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize="medium")
legend.set_bbox_to_anchor((0.5, 1.15))

# Set the x-axis tick labels
ax.set_xticks(range(1, projection_years + 1))
ax.set_xticklabels(x_tick_labels_projected)

# Adjust the layout of the plot to make room for the legend
plt.tight_layout()


# Historic growth plot
historic_year_plot = range(1, 6)
projected_year_plot = range(1, 6)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(historic_year_plot, revenue_history, label="H_Revenue")
ax.plot(historic_year_plot, cost_of_revenue_history, label="H_Cost of revenue")
ax.plot(historic_year_plot, operating_expenses_history, label="H_Operating expenses")
ax.plot(historic_year_plot, other_income_history, label="H_Other income")
ax.plot(historic_year_plot, interest_expense_history, label="H_Interest expense")
ax.plot(historic_year_plot, income_tax_history, label="H_Income tax")
# ax.plot(historic_year_plot, net_income_array, label='H_Net Income')
ax.plot(historic_year_plot, net_income_array, label="H_Net Income")
ax.plot(historic_year_plot, capex_history, label="H_Capital expenditures")
ax.plot(historic_year_plot, depreciation_history, label="H_Depreciation & amortization")
ax.plot(historic_year_plot, current_assets_history, label="H_Current assets")
ax.plot(historic_year_plot, current_liabilities_history, label="H_Current liabilities")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1000000000000, decimals=2))
ax.yaxis.set_major_locator(plt.MaxNLocator(20))
ax.set_xlabel("Year")
ax.set_ylabel("Amount")
# Set the position of the title
ax.set_title("Historic Growth", y=1)

# Set the position of the legend
legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize="medium")
legend.set_bbox_to_anchor((0.5, 1.15))

# Set the x-axis tick labels
ax.set_xticks(years_back)
ax.set_xticklabels(x_tick_labels)

# Adjust the layout of the plot to make room for the legend
plt.tight_layout()

# Average growth plot
fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(historic_year_plot, [revenue_average_yoy_growth] * len(historic_year_plot), label='A_Revenue')
ax.plot(
    historic_year_plot,
    [revenue_weighted_average_yoy_growth] * len(historic_year_plot),
    label="A_Revenue",
)
# ax.plot(historic_year_plot, revenue_average_yoy_growth, label='Revenue')
ax.plot(
    historic_year_plot,
    [cost_of_revenue_weighted_average_yoy_growth] * len(historic_year_plot),
    label="A_Cost of revenue",
)
# ax.plot(historic_year_plot, cost_of_revenue_weighted_average_yoy_growth, label='Cost of revenue')
ax.plot(
    historic_year_plot,
    [operating_expenses_weighted_average_yoy_growth] * len(historic_year_plot),
    label="A_Operating expenses",
)
# ax.plot(historic_year_plot, operating_expenses_weighted_average_yoy_growth, label='Operating expenses')
ax.plot(
    historic_year_plot,
    [other_income_weighted_average_yoy_growth] * len(historic_year_plot),
    label="A_Other income",
)
# ax.plot(historic_year_plot, other_income_weighted_average_yoy_growth, label='Other income')
ax.plot(
    historic_year_plot,
    [interest_expense_weighted_average_yoy_growth] * len(historic_year_plot),
    label="A_Interest expense",
)
# ax.plot(historic_year_plot, interest_expense_weighted_average_yoy_growth, label='Interest expense')
ax.plot(
    historic_year_plot,
    [income_tax_average_yoy_growth] * len(historic_year_plot),
    label="A_Income tax",
)
# ax.plot(historic_year_plot, income_tax_average_yoy_growth, label='Income tax')
# ax.plot(historic_year_plot, [net_income_yoy_average_growth] * len(historic_year_plot), label='A_Net Income')
ax.plot(
    historic_year_plot,
    [net_income_weighted_average_yoy_growth] * len(historic_year_plot),
    label="A_Net Income",
)
ax.plot(
    historic_year_plot,
    [capex_weighted_average_yoy_growth] * len(historic_year_plot),
    label="A_Capital expenditures",
)
# ax.plot(historic_year_plot, capex_weighted_average_yoy_growth, label='Capital expenditures')
ax.plot(
    historic_year_plot,
    [depreciation_weighted_average_yoy_growth] * len(historic_year_plot),
    label="A_Depreciation & amortization",
)
# ax.plot(historic_year_plot, depreciation_weighted_average_yoy_growth, label='Depreciation & amortization')
ax.plot(
    historic_year_plot,
    [current_assets_weighted_average_yoy_growth] * len(historic_year_plot),
    label="A_Current assets",
)
# ax.plot(historic_year_plot, current_assets_weighted_average_yoy_growth, label='Current assets')
ax.plot(
    historic_year_plot,
    [current_liabilities_weighted_average_yoy_growth] * len(historic_year_plot),
    label="A_Current liabilities",
)
# ax.plot(historic_year_plot, current_liabilities_weighted_average_yoy_growth, label='Current liabilities')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=2))
ax.yaxis.set_major_locator(plt.MaxNLocator(20))
ax.set_xlabel("Year")
ax.set_ylabel("Amount")
# Set the position of the title
ax.set_title("Average Growth", y=1)

# Set the position of the legend
legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize="medium")
legend.set_bbox_to_anchor((0.5, 1.15))

# Set the x-axis tick labels
ax.set_xticks(years_back)
ax.set_xticklabels(x_tick_labels)

# Adjust the layout of the plot to make room for the legend
plt.tight_layout()

# Projected growth plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(projected_year_plot, projected_revenue, label="P_Revenue")
ax.plot(projected_year_plot, projected_cost_of_revenue, label="P_Cost of revenue")
ax.plot(projected_year_plot, projected_operating_expenses, label="P_Operating expenses")
ax.plot(projected_year_plot, projected_other_income, label="P_Other income")
ax.plot(projected_year_plot, projected_interest_expense, label="P_Interest expense")
ax.plot(projected_year_plot, projected_income_tax, label="P_Income tax")
ax.plot(projected_year_plot, yoy_projected_net_income, label="P_Net Income")
ax.plot(projected_year_plot, projected_capex, label="P_Capital expenditures")
ax.plot(
    projected_year_plot, projected_depreciation_amortization, label="P_Depreciation & amortization"
)
ax.plot(projected_year_plot, projected_current_assets, label="P_Current assets")
ax.plot(projected_year_plot, projected_current_liabilities, label="P_Current liabilities")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1000000000000, decimals=2))
ax.yaxis.set_major_locator(plt.MaxNLocator(20))
ax.set_xlabel("Year")
ax.set_ylabel("Amount")
# Set the position of the title
ax.set_title("Projected Growth", y=1)

# Set the position of the legend
legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize="medium")
legend.set_bbox_to_anchor((0.5, 1.15))

# Set the x-axis tick labels
ax.set_xticks(range(1, projection_years + 1))
ax.set_xticklabels(x_tick_labels_projected)

# Adjust the layout of the plot to make room for the legend
plt.tight_layout()

plt.show()
# ------------------------------------------------------------------------------------------------------------------------------
# save the arrays in a csv file
# create a dataframe with the arrays as columns

projections_from_statements = pd.DataFrame(
    {
        "p_revenues": p_revenues,
        "p_cost_of_goods_solds": p_cost_of_goods_solds,
        "p_gross_profits": p_gross_profits,
        "p_operating_expenseses": p_operating_expenseses,
        "p_operating_incomes": p_operating_incomes,
        "p_other_incomes": p_other_incomes,
        "p_interest_expenses": p_interest_expenses,
        "p_income_before_taxes": p_income_before_taxes,
        "p_income_taxes": p_income_taxes,
        "p_current_assetses": p_current_assetses,
        "p_current_liabilitiess": p_current_liabilitiess,
        "p_working_capitals": p_working_capitals,
        "p_capexes": p_capexes,
        "p_depreciations_amortizations": p_depreciations_amortizations,
    }
)

income_statement_projections = pd.DataFrame(
    {
        "p_revenues": p_revenues,
        "p_cost_of_goods_solds": p_cost_of_goods_solds,
        "p_gross_profits": p_gross_profits,
        "p_operating_expenseses": p_operating_expenseses,
        "p_operating_incomes": p_operating_incomes,
        "p_other_incomes": p_other_incomes,
        "p_interest_expenses": p_interest_expenses,
        "p_income_before_taxes": p_income_before_taxes,
        "p_income_taxes": p_income_taxes,
    }
)

net_income__projections = pd.DataFrame({"projected_net_income": net_income_array})

balance_sheet_projections = pd.DataFrame(
    {
        "p_current_assetses": p_current_assetses,
        "p_current_liabilitiess": p_current_liabilitiess,
        "p_working_capitals": p_working_capitals,
    }
)

cash_flow_statement_projections = pd.DataFrame(
    {"p_capexes": p_capexes, "p_depreciations_amortizations": p_depreciations_amortizations}
)

cash_flow_projections = pd.DataFrame(
    {"future_cash_flows": future_cash_flows, "discounted_cash_flows": discounted_cash_flows}
)

# save the dataframes to a CSV file
projections_from_statements.to_csv("projections_from_statements.csv", index=False)
income_statement_projections.to_csv("income_statement_projections.csv", index=False)
net_income__projections.to_csv("net_income__projections.csv", index=False)
balance_sheet_projections.to_csv("balance_sheet_projections.csv", index=False)
cash_flow_statement_projections.to_csv("cash_flow_statement_projections.csv", index=False)
cash_flow_projections.to_csv("cash_flow_projections.csv", index=False)
