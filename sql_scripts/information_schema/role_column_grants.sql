CREATE VIEW information_schema.role_column_grants
			(grantor, grantee, table_catalog, table_schema, table_name, column_name, privilege_type, is_grantable) AS
SELECT grantor,
       grantee,
       table_catalog,
       table_schema,
       table_name,
       column_name,
       privilege_type,
       is_grantable
FROM information_schema.column_privileges
WHERE (grantor::name IN (SELECT enabled_roles.role_name FROM information_schema.enabled_roles))
   OR (grantee::name IN (SELECT enabled_roles.role_name FROM information_schema.enabled_roles));

ALTER TABLE information_schema.role_column_grants
	OWNER TO postgres;

GRANT SELECT ON information_schema.role_column_grants TO PUBLIC;