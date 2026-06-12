CREATE VIEW information_schema.data_type_privileges
			(object_catalog, object_schema, object_name, object_type, dtd_identifier) AS
-- missing source code
;

ALTER TABLE information_schema.data_type_privileges
	OWNER TO postgres;

GRANT SELECT ON information_schema.data_type_privileges TO PUBLIC;