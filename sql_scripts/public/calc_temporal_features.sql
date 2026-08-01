CREATE FUNCTION public.calc_temporal_features(p_isin text default NULL::text)
	RETURNS table("isin" text, "fiscal_quarter" integer, "fiscal_month" integer, "fiscal_year" integer, "days_to_earnings" integer, "earnings_report_recency" integer, "reporting_lag" numeric, "fiscal_year_progress" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_temporal_features(text) OWNER TO postgres;