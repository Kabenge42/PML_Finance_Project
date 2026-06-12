CREATE VIEW information_schema.sequences
			(sequence_catalog, sequence_schema, sequence_name, data_type, numeric_precision, numeric_precision_radix,
			 numeric_scale, start_value, minimum_value, maximum_value, increment, cycle_option)
AS
-- missing source code
;

ALTER TABLE information_schema.sequences
	OWNER TO postgres;

GRANT SELECT ON information_schema.sequences TO PUBLIC;