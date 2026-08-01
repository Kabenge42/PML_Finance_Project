CREATE FUNCTION public.calc_eps_trajectory_features(p_isin text default NULL::text)
	RETURNS table("isin" text, "eps_qoq_growth" numeric, "eps_yoy_quarterly" numeric, "eps_positive_streak" integer, "eps_cagr_3y" numeric, "eps_cagr_5y" numeric, "eps_growth_accel" numeric, "eps_vs_5y_avg" numeric, "eps_improvement_count" integer, "eps_trajectory_score" numeric, "eps_stability" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_eps_trajectory_features(text) OWNER TO postgres;