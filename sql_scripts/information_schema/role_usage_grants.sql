CREATE VIEW information_schema.role_usage_grants
			(grantor, grantee, object_catalog, object_schema, object_name, object_type, privilege_type, is_grantable) AS
-- missing source code
;

ALTER TABLE information_schema.role_usage_grants
	OWNER TO postgres;

GRANT SELECT ON information_schema.role_usage_grants TO PUBLIC;