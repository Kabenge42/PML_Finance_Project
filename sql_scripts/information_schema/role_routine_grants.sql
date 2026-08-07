CREATE VIEW information_schema.role_routine_grants
			(grantor, grantee, specific_catalog, specific_schema, specific_name, routine_catalog, routine_schema,
			 routine_name, privilege_type, is_grantable)
AS
SELECT grantor,
       grantee,
       specific_catalog,
       specific_schema,
       specific_name,
       routine_catalog,
       routine_schema,
       routine_name,
       privilege_type,
       is_grantable
FROM information_schema.routine_privileges
WHERE (grantor::name IN (SELECT enabled_roles.role_name FROM information_schema.enabled_roles))
   OR (grantee::name IN (SELECT enabled_roles.role_name FROM information_schema.enabled_roles));

ALTER TABLE information_schema.role_routine_grants
	OWNER TO postgres;

GRANT SELECT ON information_schema.role_routine_grants TO PUBLIC;