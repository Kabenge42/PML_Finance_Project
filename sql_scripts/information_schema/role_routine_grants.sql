CREATE VIEW information_schema.role_routine_grants
			(grantor, grantee, specific_catalog, specific_schema, specific_name, routine_catalog, routine_schema,
			 routine_name, privilege_type, is_grantable)
AS
-- missing source code
;

ALTER TABLE information_schema.role_routine_grants
	OWNER TO postgres;

GRANT SELECT ON information_schema.role_routine_grants TO PUBLIC;