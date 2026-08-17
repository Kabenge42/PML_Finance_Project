create function public.calc_beta_risk_features(p_isin text default NULL::text)
	returns table("isin" text, "beta_1y" numeric, "beta_5y" numeric, "beta_spread" numeric, "beta_trend" numeric, "high_beta_flag" integer, "low_beta_flag" integer, "beta_stability_score" numeric)
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

alter function public.calc_beta_risk_features(text) owner to postgres
;