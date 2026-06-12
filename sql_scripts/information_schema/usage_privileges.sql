CREATE VIEW information_schema.usage_privileges
			(grantor, grantee, object_catalog, object_schema, object_name, object_type, privilege_type, is_grantable) AS
-- missing source code
;

ALTER TABLE information_schema.usage_privileges
	OWNER TO postgres;

GRANT SELECT ON information_schema.usage_privileges TO PUBLIC;