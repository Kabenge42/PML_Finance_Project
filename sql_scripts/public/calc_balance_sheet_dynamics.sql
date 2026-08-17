create function public.calc_balance_sheet_dynamics(p_isin text default NULL::text)
	returns table("isin" text, "cash_to_assets_pct" numeric, "cash_change_qoq" numeric, "cash_vs_5y_avg" numeric, "inventory_change_yoy" numeric, "inventory_vs_5y_avg" numeric, "receivables_change_yoy" numeric, "receivables_vs_5y_avg" numeric, "working_capital_vs_5y_avg" numeric, "retained_earnings_vs_5y" numeric, "intangibles_growth_flag" integer, "asset_quality_score" numeric, "balance_sheet_strength" numeric, "debt_maturity_risk" numeric)
	stable
	parallel safe
	language sql
as
$$
	begin
-- missing source code
end;
$$
;

alter function public.calc_balance_sheet_dynamics(text) owner to postgres
;