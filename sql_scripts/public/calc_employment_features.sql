create function public.calc_employment_features(p_isin text default NULL::text)
	returns table("isin" text, "revenue_per_employee" numeric, "profit_per_employee" numeric, "ebitda_per_employee" numeric, "assets_per_employee" numeric, "fte_growth_1y_pct" numeric, "fte_growth_3y_pct" numeric, "workforce_stability" numeric)
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

alter function public.calc_employment_features(text) owner to postgres
;