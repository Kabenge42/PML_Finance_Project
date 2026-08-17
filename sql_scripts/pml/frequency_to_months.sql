create function frequency_to_months(earnings_report_frequency text, fy_end_date date default NULL::date, next_fy_end_date date default NULL::date) returns integer
	immutable
	language plpgsql
as
$$
begin
	-- missing source code
end;
$$
;

alter function frequency_to_months(text, date, date) owner to postgres
;