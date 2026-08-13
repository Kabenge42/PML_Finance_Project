create function pml.target_drift(arr numeric[]) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT AVG(pml.calc_change_ratio(arr[i], arr[i + 1]))
FROM generate_subscripts(arr, 1) AS i
WHERE i < array_length(arr, 1);
$$
;

alter function pml.target_drift(numeric[]) owner to postgres
;

create function pml.target_drift(arr double precision[]) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT AVG(pml.calc_change_ratio(arr[i], arr[i + 1]))
FROM generate_subscripts(arr, 1) AS i
WHERE i < array_length(arr, 1);
$$
;

alter function pml.target_drift(double precision[]) owner to postgres
;

create function pml.target_drift(arr numeric[], min_points integer) returns numeric
	immutable
	parallel safe
	language sql
as
$$
SELECT CASE WHEN pml.target_drift_n(arr) >= min_points THEN pml.target_drift(arr) END;
$$
;

alter function pml.target_drift(numeric[], integer) owner to postgres
;

create function pml.target_drift(arr double precision[], min_points integer) returns double precision
	immutable
	parallel safe
	language sql
as
$$
SELECT CASE WHEN pml.target_drift_n(arr) >= min_points THEN pml.target_drift(arr) END;
$$
;

alter function pml.target_drift(double precision[], integer) owner to postgres
;