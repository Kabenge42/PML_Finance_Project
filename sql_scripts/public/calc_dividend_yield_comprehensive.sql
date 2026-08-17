create function public.calc_dividend_yield_comprehensive(p_isin text default NULL::text)
	returns table("isin" text, "div_yield_ltm" numeric, "div_yield_ntm" numeric, "div_yield_ind" numeric, "div_yield_1fy_ind" numeric, "div_yield_5y_avg" numeric, "div_yield_vs_5y_avg" numeric, "div_yield_growth_expected" numeric, "dividend_streak" integer, "high_yield_flag" integer, "sustainable_dividend_flag" integer)
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

alter function public.calc_dividend_yield_comprehensive(text) owner to postgres
;