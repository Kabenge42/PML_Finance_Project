CREATE VIEW information_schema.column_privileges
			(grantor, grantee, table_catalog, table_schema, table_name, column_name, privilege_type, is_grantable) AS
-- missing source code
;

ALTER TABLE information_schema.column_privileges
	OWNER TO postgres;

GRANT SELECT ON information_schema.column_privileges TO PUBLIC;