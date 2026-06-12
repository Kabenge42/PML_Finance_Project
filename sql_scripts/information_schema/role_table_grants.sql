CREATE VIEW information_schema.role_table_grants
			(grantor, grantee, table_catalog, table_schema, table_name, privilege_type, is_grantable, with_hierarchy) AS
-- missing source code
;

ALTER TABLE information_schema.role_table_grants
	OWNER TO postgres;

GRANT SELECT ON information_schema.role_table_grants TO PUBLIC;