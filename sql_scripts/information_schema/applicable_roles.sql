CREATE VIEW information_schema.applicable_roles(grantee, role_name, is_grantable) AS
SELECT a.rolname::information_schema.sql_identifier                                                AS grantee,
       b.rolname::information_schema.sql_identifier                                                AS role_name,
       CASE WHEN m.admin_option THEN 'YES'::text ELSE 'NO'::text END::information_schema.yes_or_no AS is_grantable
FROM (SELECT pg_auth_members.member, pg_auth_members.roleid, pg_auth_members.admin_option
      FROM pg_auth_members
      UNION
      SELECT pg_database.datdba, pg_authid.oid, FALSE
      FROM pg_database,
           pg_authid
      WHERE pg_database.datname = current_database()
	    AND pg_authid.rolname = 'pg_database_owner'::name
     )                  m
	     JOIN pg_authid a ON m.member = a.oid
	     JOIN pg_authid b ON m.roleid = b.oid
WHERE pg_has_role(a.oid, 'USAGE'::text);

ALTER TABLE information_schema.applicable_roles
	OWNER TO postgres;

GRANT SELECT ON information_schema.applicable_roles TO PUBLIC;