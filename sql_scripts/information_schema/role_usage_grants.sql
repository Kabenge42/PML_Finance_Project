CREATE VIEW information_schema.role_usage_grants
			(grantor, grantee, object_catalog, object_schema, object_name, object_type, privilege_type, is_grantable) AS
SELECT grantor,
       grantee,
       object_catalog,
       object_schema,
       object_name,
       object_type,
       privilege_type,
       is_grantable
FROM information_schema.usage_privileges
WHERE (grantor::name IN (SELECT enabled_roles.role_name FROM information_schema.enabled_roles))
   OR (grantee::name IN (SELECT enabled_roles.role_name FROM information_schema.enabled_roles));

ALTER TABLE information_schema.role_usage_grants
	OWNER TO postgres;

GRANT SELECT ON information_schema.role_usage_grants TO PUBLIC;