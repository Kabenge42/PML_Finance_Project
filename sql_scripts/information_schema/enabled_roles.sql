CREATE VIEW information_schema.enabled_roles(role_name) AS
SELECT rolname::information_schema.sql_identifier AS role_name
FROM pg_authid a
WHERE pg_has_role(oid, 'USAGE'::text);

ALTER TABLE information_schema.enabled_roles
	OWNER TO postgres;

GRANT SELECT ON information_schema.enabled_roles TO PUBLIC;