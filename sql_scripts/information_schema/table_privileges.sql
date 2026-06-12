CREATE VIEW information_schema.table_privileges
			(grantor, grantee, table_catalog, table_schema, table_name, privilege_type, is_grantable, with_hierarchy) AS
-- missing source code
;

ALTER TABLE information_schema.table_privileges
	OWNER TO postgres;

GRANT SELECT ON information_schema.table_privileges TO PUBLIC;