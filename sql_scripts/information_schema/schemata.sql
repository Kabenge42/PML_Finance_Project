create view information_schema.schemata
			(catalog_name, schema_name, schema_owner, default_character_set_catalog, default_character_set_schema,
			 default_character_set_name, sql_path)
as
-- missing source code
;

alter table information_schema.schemata
	owner to postgres
;

grant select on information_schema.schemata to public
;