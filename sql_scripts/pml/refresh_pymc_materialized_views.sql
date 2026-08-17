create procedure refresh_pymc_materialized_views(use_concurrently boolean default true, assert_coverage boolean default false)
	language plpgsql
as
$$
begin
	-- missing source code
end;
$$
;

alter procedure refresh_pymc_materialized_views(boolean, boolean) owner to postgres
;