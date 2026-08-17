create function public.calc_valuation_features(p_isin text default NULL::text)
	returns table("isin" text, "p_e_ratio" numeric, "p_b_ratio" numeric, "ev_ebitda_ratio" numeric, "ev_sales_ratio" numeric, "dividend_yield" numeric, "peg_ratio" numeric)
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

alter function public.calc_valuation_features(text) owner to postgres
;