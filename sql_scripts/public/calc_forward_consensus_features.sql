create function public.calc_forward_consensus_features(p_isin text default NULL::text)
	returns table("isin" text, "pe_ntm" numeric, "pe_est_fy1" numeric, "pe_forward_discount" numeric, "eps_gaap_vs_norm_ntm" numeric, "eps_gaap_vs_norm_fy1e" numeric, "forward_adjustment_trend" numeric, "ebitda_est_ntm" numeric, "ebitda_est_fy1e" numeric, "ev_ebitda_est_fy1" numeric, "ebitda_forward_growth" numeric, "earnings_revision_divergence" numeric, "forward_pe_vs_sector_proxy" numeric)
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

alter function public.calc_forward_consensus_features(text) owner to postgres
;