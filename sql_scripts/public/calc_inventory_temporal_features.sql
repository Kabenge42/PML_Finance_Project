create function public.calc_inventory_temporal_features(p_isin text default NULL::text)
	returns table("isin" text, "inventory_ltm" numeric, "inventory_fq" numeric, "inventory_fy" numeric, "inventory_1fq" numeric, "inventory_2fq" numeric, "inventory_3fq" numeric, "inventory_4fq" numeric, "inventory_1fy" numeric, "inventory_2fy" numeric, "inventory_3fy" numeric, "inventory_4fy" numeric, "inventory_qoq_change" numeric, "inventory_yoy_change" numeric, "inventory_4q_trend" numeric, "inventory_vs_5y_avg" numeric, "inventory_days" numeric, "inventory_turnover" numeric, "inventory_to_revenue" numeric, "inventory_to_assets" numeric, "inventory_buildup_flag" integer, "inventory_reduction_flag" integer, "inventory_volatility" numeric)
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

alter function public.calc_inventory_temporal_features(text) owner to postgres
;