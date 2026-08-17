create function public.calc_sentiment_features(p_isin text default NULL::text)
	returns table("isin" text, "analyst_bullish_pct" numeric, "analyst_bearish_pct" numeric, "analyst_neutral_pct" numeric, "analyst_conviction" numeric, "upside_potential" numeric, "price_target_spread_pct" numeric, "price_target_revision_1m" numeric, "price_target_revision_3m" numeric, "eps_revision_momentum" numeric, "analyst_rating_normalized" numeric, "analyst_coverage_quality" numeric)
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

alter function public.calc_sentiment_features(text) owner to postgres
;