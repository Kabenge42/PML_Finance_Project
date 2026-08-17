create function public.calc_goodwill_temporal_features(p_isin text default NULL::text)
	returns table("isin" text, "goodwill_fq" numeric, "goodwill_ltm" numeric, "goodwill_fy" numeric, "goodwill_1fq" numeric, "goodwill_2fq" numeric, "goodwill_3fq" numeric, "goodwill_4fq" numeric, "goodwill_1fy" numeric, "goodwill_2fy" numeric, "goodwill_3fy" numeric, "goodwill_4fy" numeric, "goodwill_qoq_change" numeric, "goodwill_yoy_change" numeric, "goodwill_3y_growth" numeric, "goodwill_vs_5y_avg" numeric, "recent_acquisition_flag" integer, "goodwill_accumulation_rate" numeric, "goodwill_to_assets_trend" numeric, "impairment_risk_score" numeric, "goodwill_concentration" numeric)
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

alter function public.calc_goodwill_temporal_features(text) owner to postgres
;