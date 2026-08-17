create function public.calc_cashflow_features(p_isin text default NULL::text)
	returns table("isin" text, "cfo_to_net_income" numeric, "fcf_to_net_income" numeric, "fcf_margin" numeric, "cfo_growth_yoy" numeric, "fcf_positive_ratio" numeric, "acquisition_intensity" numeric, "self_funding_ratio" numeric)
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

alter function public.calc_cashflow_features(text) owner to postgres
;