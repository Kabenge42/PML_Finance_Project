create function public.refresh_all_stock_features() returns void
	language plpgsql
as
$$
begin
	-- missing source code
end;
$$
;

comment on function public.refresh_all_stock_features() is 'Refreshes the mv_all_stock_features materialized view concurrently (non-blocking).
    Call periodically after equities table updates.'
;

alter function public.refresh_all_stock_features() owner to postgres
;