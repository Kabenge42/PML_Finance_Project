CREATE VIEW information_schema.enabled_roles(role_name) AS
-- missing source code
;

ALTER TABLE information_schema.enabled_roles
	OWNER TO postgres;

GRANT SELECT ON information_schema.enabled_roles TO PUBLIC;