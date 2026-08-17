create function public.calc_total_debt_temporal(p_isin text default NULL::text)
	returns table("isin" text, "debt_fq" numeric, "debt_fy" numeric, "debt_ltm" numeric, "debt_1fq" numeric, "debt_2fq" numeric, "debt_3fq" numeric, "debt_4fq" numeric, "debt_1fy" numeric, "debt_2fy" numeric, "debt_3fy" numeric, "debt_4fy" numeric, "debt_qoq_change" numeric, "debt_yoy_change" numeric, "debt_4q_trend" numeric, "debt_3y_cagr" numeric, "debt_deleveraging" integer, "debt_to_equity_trend" numeric)
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

alter function public.calc_total_debt_temporal(text) owner to postgres
;