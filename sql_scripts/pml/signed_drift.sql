create function signed_drift(arr numeric[]) returns numeric
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

alter function signed_drift(numeric[]) owner to postgres
;

create function signed_drift(arr double precision[]) returns double precision
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

alter function signed_drift(double precision[]) owner to postgres
;

create function signed_drift(arr numeric[], min_points integer) returns numeric
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

alter function signed_drift(numeric[], integer) owner to postgres
;

create function signed_drift(arr double precision[], min_points integer) returns double precision
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

alter function signed_drift(double precision[], integer) owner to postgres
;