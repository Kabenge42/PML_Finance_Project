create function public.calc_gross_profit_temporal(p_isin text default NULL::text)
	returns table("isin" text, "gp_fq" numeric, "gp_fy" numeric, "gp_ltm" numeric, "gp_1fqfq" numeric, "gp_2fqfq" numeric, "gp_3fqfq" numeric, "gp_4fqfq" numeric, "gp_1fy" numeric, "gp_2fy" numeric, "gp_3fy" numeric, "gp_4fy" numeric, "gp_qoq_growth" numeric, "gp_yoy_growth" numeric, "gp_margin_fq" numeric, "gp_margin_trend" numeric, "gp_positive_quarters" integer, "gp_margin_expansion" integer)
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

alter function public.calc_gross_profit_temporal(text) owner to postgres
;