CREATE VIEW information_schema.udt_privileges
			(grantor, grantee, udt_catalog, udt_schema, udt_name, privilege_type, is_grantable) AS
-- missing source code
;

ALTER TABLE information_schema.udt_privileges
	OWNER TO postgres;

GRANT SELECT ON information_schema.udt_privileges TO PUBLIC;