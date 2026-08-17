create function public.calc_investment_income_temporal(p_isin text default NULL::text)
	returns table("isin" text, "inv_income_ltm" numeric, "inv_income_fq" numeric, "inv_income_fy" numeric, "inv_income_qoq_growth" numeric, "inv_income_yoy_growth" numeric, "inv_income_to_revenue" numeric, "inv_income_trend_3y" numeric, "inv_income_positive_quarters" integer, "financial_company_proxy" integer)
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

alter function public.calc_investment_income_temporal(text) owner to postgres
;