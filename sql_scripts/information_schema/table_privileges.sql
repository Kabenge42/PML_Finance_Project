CREATE VIEW information_schema.table_privileges
			(grantor, grantee, table_catalog, table_schema, table_name, privilege_type, is_grantable, with_hierarchy) AS
SELECT u_grantor.rolname::information_schema.sql_identifier                                                   AS grantor,
       grantee.rolname::information_schema.sql_identifier                                                     AS grantee,
       current_database()::information_schema.sql_identifier                                                  AS table_catalog,
       nc.nspname::information_schema.sql_identifier                                                          AS table_schema,
       c.relname::information_schema.sql_identifier                                                           AS table_name,
       c.prtype::information_schema.character_data                                                            AS privilege_type,
       CASE
	       WHEN pg_has_role(grantee.oid, c.relowner, 'USAGE'::text) OR c.grantable THEN 'YES'::text
	       ELSE 'NO'::text END::information_schema.yes_or_no                                                  AS is_grantable,
       CASE
	       WHEN c.prtype = 'SELECT'::text THEN 'YES'::text
	       ELSE 'NO'::text END::information_schema.yes_or_no                                                  AS with_hierarchy
FROM (SELECT pg_class.oid,
             pg_class.relname,
             pg_class.relnamespace,
             pg_class.relkind,
             pg_class.relowner,
             (aclexplode(COALESCE(pg_class.relacl, acldefault('r'::"char", pg_class.relowner)))).grantor        AS grantor,
             (aclexplode(COALESCE(pg_class.relacl,
                                  acldefault('r'::"char", pg_class.relowner)))).grantee                         AS grantee,
             (aclexplode(COALESCE(pg_class.relacl,
                                  acldefault('r'::"char", pg_class.relowner)))).privilege_type                  AS privilege_type,
             (aclexplode(COALESCE(pg_class.relacl,
                                  acldefault('r'::"char", pg_class.relowner)))).is_grantable                    AS is_grantable
      FROM pg_class
     )                                                                                                       c(oid,
                                                                                                               relname,
                                                                                                               relnamespace,
                                                                                                               relkind,
                                                                                                               relowner,
                                                                                                               grantor,
                                                                                                               grantee,
                                                                                                               prtype,
                                                                                                               grantable),
     pg_namespace                                                                                            nc,
     pg_authid                                                                                               u_grantor,
     (SELECT pg_authid.oid, pg_authid.rolname FROM pg_authid UNION ALL SELECT 0::oid AS oid, 'PUBLIC'::name
     )                                                                                                       grantee(oid, rolname)
WHERE c.relnamespace = nc.oid
  AND (c.relkind = ANY (ARRAY ['r'::"char", 'v'::"char", 'f'::"char", 'p'::"char"]))
  AND c.grantee = grantee.oid
  AND c.grantor = u_grantor.oid
  AND (c.prtype = ANY
       (ARRAY ['INSERT'::text, 'SELECT'::text, 'UPDATE'::text, 'DELETE'::text, 'TRUNCATE'::text, 'REFERENCES'::text, 'TRIGGER'::text]))
  AND (pg_has_role(u_grantor.oid, 'USAGE'::text) OR pg_has_role(grantee.oid, 'USAGE'::text) OR
       grantee.rolname = 'PUBLIC'::name);

ALTER TABLE information_schema.table_privileges
	OWNER TO postgres;

GRANT SELECT ON information_schema.table_privileges TO PUBLIC;