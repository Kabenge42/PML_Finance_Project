CREATE FUNCTION public.calc_goodwill_temporal_features(p_isin text default NULL::text)
	RETURNS table("isin" text, "goodwill_fq" numeric, "goodwill_ltm" numeric, "goodwill_fy" numeric, "goodwill_1fq" numeric, "goodwill_2fq" numeric, "goodwill_3fq" numeric, "goodwill_4fq" numeric, "goodwill_1fy" numeric, "goodwill_2fy" numeric, "goodwill_3fy" numeric, "goodwill_4fy" numeric, "goodwill_qoq_change" numeric, "goodwill_yoy_change" numeric, "goodwill_3y_growth" numeric, "goodwill_vs_5y_avg" numeric, "recent_acquisition_flag" integer, "goodwill_accumulation_rate" numeric, "goodwill_to_assets_trend" numeric, "impairment_risk_score" numeric, "goodwill_concentration" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_goodwill_temporal_features(text) OWNER TO postgres;