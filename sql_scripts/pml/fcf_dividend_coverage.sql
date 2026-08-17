create function fcf_dividend_coverage(fcf numeric, dividends_paid numeric) returns numeric
	immutable
	parallel safe
	language sql
as
$$
	begin
-- missing source code
end;
$$
;

alter function fcf_dividend_coverage(numeric, numeric) owner to postgres
;

create function fcf_dividend_coverage(fcf double precision, dividends_paid double precision) returns double precision
	immutable
	parallel safe
	language sql
as
$$
	begin
-- missing source code
end;
$$
;

alter function fcf_dividend_coverage(double precision, double precision) owner to postgres
;