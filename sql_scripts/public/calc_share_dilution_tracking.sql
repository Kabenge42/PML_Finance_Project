create function public.calc_share_dilution_tracking(p_isin text default NULL::text)
	returns table("isin" text, "shrs_out_1fy" numeric, "shares_yoy_change_pct" numeric, "net_buyback_flag" integer)
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

alter function public.calc_share_dilution_tracking(text) owner to postgres
;