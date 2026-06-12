CREATE FUNCTION public.calc_revenue_estimate_consensus(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                       text,
		        "revenue_est_avg_fy1e"       numeric,
		        "revenue_est_med_fy1e"       numeric,
		        "revenue_est_avg_ntm"        numeric,
		        "revenue_est_med_ntm"        numeric,
		        "revenue_avg_med_diff_pct"   numeric,
		        "revenue_consensus_strength" numeric,
		        "revenue_revision_trend"     numeric,
		        "revenue_vs_current"         numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_revenue_estimate_consensus(text) OWNER TO postgres;