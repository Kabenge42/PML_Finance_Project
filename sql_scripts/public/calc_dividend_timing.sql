create function public.calc_dividend_timing(p_isin text default NULL::text)
	returns table("isin" text, "days_since_ex_date" integer, "days_to_payment" integer, "dividend_announced_flag" integer, "ex_date_approaching_flag" integer, "dividend_frequency_score" integer, "dividend_consistency" numeric, "recent_dividend_change" numeric, "dividend_yield_vs_5y_avg" numeric)
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

alter function public.calc_dividend_timing(text) owner to postgres
;