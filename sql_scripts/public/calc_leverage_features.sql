create function public.calc_leverage_features(p_isin text default NULL::text)
	returns table("isin" text, "debt_to_equity" numeric, "debt_to_assets" numeric, "equity_ratio" numeric, "interest_coverage" numeric, "current_ratio" numeric, "cash_ratio" numeric, "working_capital_ratio" numeric)
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

alter function public.calc_leverage_features(text) owner to postgres
;