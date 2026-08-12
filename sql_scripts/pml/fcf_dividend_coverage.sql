-- Cyclic dependencies found

create function fcf_dividend_coverage(fcf double precision, dividends_paid double precision) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT pml.safe_divide(fcf, ABS(dividends_paid));
$$
;

alter function fcf_dividend_coverage(unknown, unknown) owner to postgres
;

create function fcf_dividend_coverage(fcf numeric, dividends_paid numeric) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT pml.safe_divide(fcf, ABS(dividends_paid));
$$
;

alter function fcf_dividend_coverage(unknown, unknown) owner to postgres
;