create function public.calc_eps_continuing_features(p_isin text default NULL::text)
	returns table("isin" text, "eps_cont_ltm" numeric, "eps_cont_fq" numeric, "eps_cont_fy" numeric, "eps_cont_1fqfq" numeric, "eps_cont_2fqfq" numeric, "eps_cont_3fqfq" numeric, "eps_cont_4fqfq" numeric, "eps_cont_1fy" numeric, "eps_cont_2fy" numeric, "eps_cont_3fy" numeric, "eps_cont_4fy" numeric, "eps_cont_qoq_growth" numeric, "eps_cont_yoy_growth" numeric, "eps_cont_cagr_3y" numeric, "eps_cont_vs_total_eps" numeric, "eps_cont_positive_streak" integer, "eps_cont_trajectory_score" numeric, "discontinued_ops_impact" numeric, "core_earnings_stability" numeric)
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

alter function public.calc_eps_continuing_features(text) owner to postgres
;