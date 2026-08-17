create function public.calc_price_target_dynamics(p_isin text default NULL::text)
	returns table("isin" text, "pt_momentum_1w" numeric, "pt_momentum_1m" numeric, "pt_momentum_3m" numeric, "pt_momentum_6m" numeric, "pt_momentum_1y" numeric, "pt_median_momentum_1m" numeric, "pt_median_momentum_3m" numeric, "pt_acceleration_short" numeric, "pt_acceleration_long" numeric, "pt_consensus_convergence" numeric, "analyst_coverage_change_1m" integer, "analyst_coverage_change_3m" integer, "analyst_coverage_change_1y" integer, "pt_vs_price_momentum" numeric, "analyst_coverage_trend" numeric)
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

alter function public.calc_price_target_dynamics(text) owner to postgres
;