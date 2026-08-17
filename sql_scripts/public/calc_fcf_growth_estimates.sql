create function public.calc_fcf_growth_estimates(p_isin text default NULL::text)
	returns table("isin" text, "fcf_est_fy1" numeric, "fcf_est_fy2" numeric, "fcf_est_fy3" numeric, "fcf_est_fy4" numeric, "fcf_est_fy5" numeric, "fcf_est_growth_fy1_vs_ltm" numeric, "fcf_est_growth_fy2_vs_fy1" numeric, "fcf_est_growth_fy3_vs_fy2" numeric, "fcf_est_growth_fy4_vs_fy3" numeric, "fcf_est_growth_fy5_vs_fy4" numeric, "fcf_est_cagr_3y" numeric, "fcf_est_cagr_5y" numeric, "fcf_est_margin_fy1" numeric, "fcf_est_yield_fy1" numeric, "fcf_est_growth_acceleration" numeric, "fcf_est_growth_deceleration" integer, "fcf_est_trajectory_score" numeric, "fcf_est_always_positive" integer, "fcf_est_vs_historical" numeric, "fcf_est_capex_implied_ratio" numeric)
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

comment on function public.calc_fcf_growth_estimates(text) is 'Estimated free cash flow growth rates from consensus FCF forecasts (FY1E-FY5E).
     Calculates YoY growth rates, 3Y/5Y CAGRs, growth acceleration, forward margins/yields,
     and trajectory quality scores. Source: FCF - Est Avg (FY1E through FY5E).'
;

alter function public.calc_fcf_growth_estimates(text) owner to postgres
;