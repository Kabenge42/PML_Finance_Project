create view information_schema.sequences
			(sequence_catalog, sequence_schema, sequence_name, data_type, numeric_precision, numeric_precision_radix,
			 numeric_scale, start_value, minimum_value, maximum_value, increment, cycle_option)
as
-- missing source code
;

alter table information_schema.sequences
	owner to postgres
;

grant select on information_schema.sequences to public
;