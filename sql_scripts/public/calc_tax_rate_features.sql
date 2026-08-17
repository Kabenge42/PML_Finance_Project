create function public.calc_tax_rate_features(p_isin text default NULL::text)
	returns table("isin" text, "effective_tax_rate_ltm" numeric, "effective_tax_rate_fy" numeric, "tax_rate_yoy_change" numeric, "tax_rate_qoq_change" numeric, "tax_rate_stability" numeric, "low_tax_flag" integer, "tax_rate_trend_4q" numeric)
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

alter function public.calc_tax_rate_features(text) owner to postgres
;