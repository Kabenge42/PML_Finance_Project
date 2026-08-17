create function public.calc_eps_trajectory_features(p_isin text default NULL::text)
	returns table("isin" text, "eps_qoq_growth" numeric, "eps_yoy_quarterly" numeric, "eps_positive_streak" integer, "eps_cagr_3y" numeric, "eps_cagr_5y" numeric, "eps_growth_accel" numeric, "eps_vs_5y_avg" numeric, "eps_improvement_count" integer, "eps_trajectory_score" numeric, "eps_stability" numeric)
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

alter function public.calc_eps_trajectory_features(text) owner to postgres
;