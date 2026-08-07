CREATE VIEW information_schema.role_udt_grants
			(grantor, grantee, udt_catalog, udt_schema, udt_name, privilege_type, is_grantable) AS
SELECT grantor, grantee, udt_catalog, udt_schema, udt_name, privilege_type, is_grantable
FROM information_schema.udt_privileges
WHERE (grantor::name IN (SELECT enabled_roles.role_name FROM information_schema.enabled_roles))
   OR (grantee::name IN (SELECT enabled_roles.role_name FROM information_schema.enabled_roles));

ALTER TABLE information_schema.role_udt_grants
	OWNER TO postgres;

GRANT SELECT ON information_schema.role_udt_grants TO PUBLIC;