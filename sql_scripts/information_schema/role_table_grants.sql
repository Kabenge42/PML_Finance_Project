CREATE VIEW information_schema.role_table_grants
			(grantor, grantee, table_catalog, table_schema, table_name, privilege_type, is_grantable, with_hierarchy) AS
SELECT grantor,
       grantee,
       table_catalog,
       table_schema,
       table_name,
       privilege_type,
       is_grantable,
       with_hierarchy
FROM information_schema.table_privileges
WHERE (grantor::name IN (SELECT enabled_roles.role_name FROM information_schema.enabled_roles))
   OR (grantee::name IN (SELECT enabled_roles.role_name FROM information_schema.enabled_roles));

ALTER TABLE information_schema.role_table_grants
	OWNER TO postgres;

GRANT SELECT ON information_schema.role_table_grants TO PUBLIC;