create function public.calc_rnd_temporal_features(p_isin text default NULL::text)
	returns table("isin" text, "rnd_ltm" numeric, "rnd_fq" numeric, "rnd_fy" numeric, "rnd_1fqfq" numeric, "rnd_2fqfq" numeric, "rnd_3fqfq" numeric, "rnd_4fqfq" numeric, "rnd_1fy" numeric, "rnd_2fy" numeric, "rnd_3fy" numeric, "rnd_4fy" numeric, "rnd_intensity_ltm" numeric, "rnd_intensity_fy" numeric, "rnd_intensity_trend" numeric, "rnd_qoq_growth" numeric, "rnd_yoy_growth" numeric, "rnd_cagr_3y" numeric, "rnd_per_employee" numeric, "rnd_to_gross_profit" numeric, "rnd_roi_proxy" numeric, "rnd_increasing_flag" integer, "rnd_cut_flag" integer, "high_rnd_intensity_flag" integer)
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

alter function public.calc_rnd_temporal_features(text) owner to postgres
;