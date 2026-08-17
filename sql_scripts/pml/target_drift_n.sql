create function target_drift_n(arr numeric[]) returns integer
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

alter function target_drift_n(numeric[]) owner to postgres
;

create function target_drift_n(arr double precision[]) returns integer
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

alter function target_drift_n(double precision[]) owner to postgres
;