CREATE FUNCTION public.calc_fcf_estimate_features(p_isin text default NULL::text)
	RETURNS table("isin" text, "fcf_est_avg_fy1e" numeric, "fcf_est_avg_fy2e" numeric, "fcf_est_avg_fy3e" numeric, "fcf_est_avg_fy4e" numeric, "fcf_est_avg_fy5e" numeric, "fcf_est_cagr_5y" numeric, "fcf_est_trend" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_fcf_estimate_features(text) OWNER TO postgres;