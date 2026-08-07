CREATE VIEW information_schema.administrable_role_authorizations(grantee, role_name, is_grantable) AS
SELECT grantee, role_name, is_grantable
FROM information_schema.applicable_roles
WHERE is_grantable::text = 'YES'::text;

ALTER TABLE information_schema.administrable_role_authorizations
	OWNER TO postgres;

GRANT SELECT ON information_schema.administrable_role_authorizations TO PUBLIC;