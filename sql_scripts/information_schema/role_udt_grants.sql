CREATE VIEW information_schema.role_udt_grants
			(grantor, grantee, udt_catalog, udt_schema, udt_name, privilege_type, is_grantable) AS
-- missing source code
;

ALTER TABLE information_schema.role_udt_grants
	OWNER TO postgres;

GRANT SELECT ON information_schema.role_udt_grants TO PUBLIC;