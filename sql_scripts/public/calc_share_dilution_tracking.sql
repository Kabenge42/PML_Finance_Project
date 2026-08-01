CREATE FUNCTION public.calc_share_dilution_tracking(p_isin text default NULL::text)
	RETURNS table("isin" text, "shrs_out_1fy" numeric, "shares_yoy_change_pct" numeric, "net_buyback_flag" integer)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_share_dilution_tracking(text) OWNER TO postgres;