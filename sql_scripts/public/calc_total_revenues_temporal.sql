CREATE FUNCTION public.calc_total_revenues_temporal(p_isin text default NULL::text)
	RETURNS table("isin" text, "revenue_fq" numeric, "revenue_ltm" numeric, "revenue_fy" numeric, "revenue_1fy" numeric, "revenue_5yavgfq" numeric, "revenue_5yavgltm" numeric, "revenue_growth_yoy" numeric, "revenue_vs_5y_avg_fq" numeric, "revenue_vs_5y_avg_ltm" numeric, "revenue_fq_vs_avg" numeric, "revenue_momentum" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_total_revenues_temporal(text) OWNER TO postgres;