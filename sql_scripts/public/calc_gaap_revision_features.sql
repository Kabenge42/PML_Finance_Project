create function public.calc_gaap_revision_features(p_isin text default NULL::text)
	returns table("isin" text, "gaap_revision_momentum" numeric, "gaap_revision_1m" numeric, "gaap_revision_3m" numeric, "gaap_revision_6m" numeric, "gaap_revision_1y" numeric, "gaap_vs_norm_revision_spread" numeric, "gaap_revision_acceleration" numeric, "gaap_positive_revision_flag" integer, "revision_quality_divergence" numeric)
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

alter function public.calc_gaap_revision_features(text) owner to postgres
;