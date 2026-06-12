CREATE VIEW information_schema.role_column_grants
			(grantor, grantee, table_catalog, table_schema, table_name, column_name, privilege_type, is_grantable) AS
-- missing source code
;

ALTER TABLE information_schema.role_column_grants
	OWNER TO postgres;

GRANT SELECT ON information_schema.role_column_grants TO PUBLIC;