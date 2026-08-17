create function public.calc_revenue_estimate_consensus(p_isin text default NULL::text)
	returns table("isin" text, "revenue_est_avg_fy1e" numeric, "revenue_est_med_fy1e" numeric, "revenue_est_avg_ntm" numeric, "revenue_est_med_ntm" numeric, "revenue_avg_med_diff_pct" numeric, "revenue_consensus_strength" numeric, "revenue_revision_trend" numeric, "revenue_vs_current" numeric)
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

alter function public.calc_revenue_estimate_consensus(text) owner to postgres
;