create function public.calc_efficiency_ratios(p_isin text default NULL::text)
	returns table("isin" text, "asset_turnover" numeric, "inventory_turnover" numeric, "receivables_days" numeric, "working_capital_turns" numeric)
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

alter function public.calc_efficiency_ratios(text) owner to postgres
;