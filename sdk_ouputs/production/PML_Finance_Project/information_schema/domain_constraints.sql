create view information_schema.domain_constraints
            (constraint_catalog, constraint_schema, constraint_name, domain_catalog, domain_schema, domain_name,
             is_deferrable, initially_deferred)
as
SELECT current_database()::sql_identifier AS constraint_catalog,
       rs.nspname::sql_identifier         AS constraint_schema,
       con.conname::sql_identifier        AS constraint_name,
       current_database()::sql_identifier AS domain_catalog,
       n.nspname::sql_identifier          AS domain_schema,
       t.typname::sql_identifier          AS domain_name,
       CASE
           WHEN con.condeferrable THEN 'YES'::text
           ELSE 'NO'::text
           END::yes_or_no                 AS is_deferrable,
       CASE
           WHEN con.condeferred THEN 'YES'::text
           ELSE 'NO'::text
           END::yes_or_no                 AS initially_deferred
FROM pg_namespace rs,
     pg_namespace n,
     pg_constraint con,
     pg_type t
WHERE rs.oid = con.connamespace
  AND n.oid = t.typnamespace
  AND t.oid = con.contypid
  AND (pg_has_role(t.typowner, 'USAGE'::text) OR has_type_privilege(t.oid, 'USAGE'::text));

alter table information_schema.domain_constraints
    owner to postgres;

grant select on information_schema.domain_constraints to public;

