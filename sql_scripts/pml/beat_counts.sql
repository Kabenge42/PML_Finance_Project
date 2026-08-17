create function beat_counts(surprises numeric[])
	returns table("n_total" integer, "n_beats" integer)
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

alter function beat_counts(numeric[]) owner to postgres
;

create function beat_counts(surprises double precision[])
	returns table("n_total" integer, "n_beats" integer)
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

alter function beat_counts(double precision[]) owner to postgres
;