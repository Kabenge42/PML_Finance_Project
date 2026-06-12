CREATE VIEW information_schema.applicable_roles(grantee, role_name, is_grantable) AS
-- missing source code
;

ALTER TABLE information_schema.applicable_roles
	OWNER TO postgres;

GRANT SELECT ON information_schema.applicable_roles TO PUBLIC;