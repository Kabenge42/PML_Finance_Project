-- Cyclic dependencies found

create function target_drift_n(arr double precision[]) returns integer
	immutable
	parallel safe
	language sql
as
$$
SELECT COUNT(*)::INT
FROM generate_subscripts(arr, 1) AS i
WHERE i < array_length(arr, 1)
  AND arr[i] IS NOT NULL
  AND arr[i + 1] IS NOT NULL
  AND arr[i + 1] <> 0;
$$
;

alter function target_drift_n(unknown) owner to postgres
;

create function target_drift_n(arr numeric[]) returns integer
	immutable
	parallel safe
	language sql
as
$$
SELECT COUNT(*)::INT
FROM generate_subscripts(arr, 1) AS i
WHERE i < array_length(arr, 1)
  AND arr[i] IS NOT NULL
  AND arr[i + 1] IS NOT NULL
  AND arr[i + 1] <> 0;
$$
;

alter function target_drift_n(unknown) owner to postgres
;