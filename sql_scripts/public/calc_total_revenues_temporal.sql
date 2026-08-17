create function public.calc_total_revenues_temporal(p_isin text default NULL::text)
	returns table("isin" text, "revenue_fq" numeric, "revenue_ltm" numeric, "revenue_fy" numeric, "revenue_1fy" numeric, "revenue_5yavgfq" numeric, "revenue_5yavgltm" numeric, "revenue_growth_yoy" numeric, "revenue_vs_5y_avg_fq" numeric, "revenue_vs_5y_avg_ltm" numeric, "revenue_fq_vs_avg" numeric, "revenue_momentum" numeric)
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

alter function public.calc_total_revenues_temporal(text) owner to postgres
;