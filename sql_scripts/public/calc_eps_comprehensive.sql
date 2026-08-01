CREATE FUNCTION public.calc_eps_comprehensive(p_isin text default NULL::text)
	RETURNS table("isin" text, "eps_basic_fq" numeric, "eps_basic_ltm" numeric, "eps_basic_fy" numeric, "eps_adj_ltm" numeric, "eps_norm_est_fy1e" numeric, "eps_growth_yoy" numeric, "eps_cagr_3y" numeric, "eps_adjustment_ratio" numeric, "eps_positive_years" integer, "eps_trajectory_score" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_eps_comprehensive(text) OWNER TO postgres;