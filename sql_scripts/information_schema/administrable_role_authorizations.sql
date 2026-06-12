CREATE VIEW information_schema.administrable_role_authorizations(grantee, role_name, is_grantable) AS
-- missing source code
;

ALTER TABLE information_schema.administrable_role_authorizations
	OWNER TO postgres;

GRANT SELECT ON information_schema.administrable_role_authorizations TO PUBLIC;