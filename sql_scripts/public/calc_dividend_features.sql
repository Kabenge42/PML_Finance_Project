create function public.calc_dividend_features(p_isin text default NULL::text)
	returns table("isin" text, "dividend_streak" integer, "dividend_yield_ltm" numeric, "dividend_yield_ntm" numeric, "dividend_payout_ratio" numeric, "fcf_dividend_coverage" numeric, "buyback_yield" numeric, "total_shareholder_yield" numeric, "dividend_growth_expectation" numeric)
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

alter function public.calc_dividend_features(text) owner to postgres
;