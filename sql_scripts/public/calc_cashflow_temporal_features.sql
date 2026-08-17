create function public.calc_cashflow_temporal_features(p_isin text default NULL::text)
	returns table("isin" text, "cfo_quarterly_trend" numeric, "cfo_yoy_quarterly" numeric, "cfi_quarterly_trend" numeric, "cff_quarterly_trend" numeric, "fcf_quarterly_trend" numeric, "cfo_positive_quarters" integer, "cfi_negative_quarters" integer, "cff_pattern_score" numeric, "cash_burn_rate" numeric, "cf_volatility_score" numeric, "operating_cf_momentum" numeric, "financing_dependency" numeric)
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

alter function public.calc_cashflow_temporal_features(text) owner to postgres
;