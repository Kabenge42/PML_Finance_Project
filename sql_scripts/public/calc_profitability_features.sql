create function public.calc_profitability_features(p_isin text default NULL::text)
	returns table("isin" text, "roe" numeric, "roa" numeric, "gross_margin_pct" numeric, "operating_margin_pct" numeric, "net_margin_pct" numeric, "ebitda_margin_pct" numeric, "roic" numeric, "rnd_intensity" numeric, "equity_multiplier" numeric)
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

alter function public.calc_profitability_features(text) owner to postgres
;