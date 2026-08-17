create view information_schema.routine_sequence_usage
			(specific_catalog, specific_schema, specific_name, routine_catalog, routine_schema, routine_name,
			 sequence_catalog, sequence_schema, sequence_name)
as
-- missing source code
;

alter table information_schema.routine_sequence_usage
	owner to postgres
;

grant select on information_schema.routine_sequence_usage to public
;