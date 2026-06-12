CREATE VIEW information_schema.routine_privileges
			(grantor, grantee, specific_catalog, specific_schema, specific_name, routine_catalog, routine_schema,
			 routine_name, privilege_type, is_grantable)
AS
-- missing source code
;

ALTER TABLE information_schema.routine_privileges
	OWNER TO postgres;

GRANT SELECT ON information_schema.routine_privileges TO PUBLIC;