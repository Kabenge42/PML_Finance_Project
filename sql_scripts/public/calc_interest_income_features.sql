create function public.calc_interest_income_features(p_isin text default NULL::text)
	returns table("isin" text, "interest_income_ltm" numeric, "interest_expense_ltm" numeric, "net_interest_income" numeric, "interest_coverage_ratio" numeric, "interest_income_to_revenue" numeric, "interest_expense_to_revenue" numeric, "net_interest_margin_proxy" numeric)
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

alter function public.calc_interest_income_features(text) owner to postgres
;