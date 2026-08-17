create function public.calc_fcf_estimate_features(p_isin text default NULL::text)
	returns table("isin" text, "fcf_est_avg_fy1e" numeric, "fcf_est_avg_fy2e" numeric, "fcf_est_avg_fy3e" numeric, "fcf_est_avg_fy4e" numeric, "fcf_est_avg_fy5e" numeric, "fcf_est_cagr_5y" numeric, "fcf_est_trend" numeric)
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

alter function public.calc_fcf_estimate_features(text) owner to postgres
;