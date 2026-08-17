create function public.calc_ebit_ebitda_comprehensive(p_isin text default NULL::text)
	returns table("isin" text, "ebit_fq" numeric, "ebit_ltm" numeric, "ebit_fy" numeric, "ebit_1fy" numeric, "ebitda_fq" numeric, "ebitda_ltm" numeric, "ebitda_fy" numeric, "ebitda_1fy" numeric, "ebit_2fy" numeric, "ebit_3fy" numeric, "ebit_4fy" numeric, "ebitda_2fy" numeric, "ebitda_3fy" numeric, "ebitda_4fy" numeric, "ebit_1fqfq" numeric, "ebit_2fqfq" numeric, "ebit_3fqfq" numeric, "ebit_4fqfq" numeric, "ebitda_1fqfq" numeric, "ebitda_2fqfq" numeric, "ebitda_3fqfq" numeric, "ebitda_4fqfq" numeric, "ebit_5yavgfq" numeric, "ebit_5yavgltm" numeric, "ebitda_5yavgfq" numeric, "ebitda_5yavgltm" numeric, "ebit_adj_fq" numeric, "ebit_adj_ltm" numeric, "ebit_adj_fy" numeric, "ebitda_adj_fq" numeric, "ebitda_adj_ltm" numeric, "ebitda_adj_fy" numeric, "ebit_growth_yoy" numeric, "ebitda_growth_yoy" numeric, "ebit_margin_ltm" numeric, "ebitda_margin_ltm" numeric, "ebit_positive_years" integer, "ebitda_positive_years" integer, "ebit_qoq_growth" numeric, "ebitda_qoq_growth" numeric, "ebit_cagr_3y" numeric, "ebitda_cagr_3y" numeric, "ebit_vs_5y_avg" numeric, "ebitda_vs_5y_avg" numeric)
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

alter function public.calc_ebit_ebitda_comprehensive(text) owner to postgres
;