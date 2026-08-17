create view information_schema.collation_character_set_applicability
			(collation_catalog, collation_schema, collation_name, character_set_catalog, character_set_schema,
			 character_set_name)
as
-- missing source code
;

alter table information_schema.collation_character_set_applicability
	owner to postgres
;

grant select on information_schema.collation_character_set_applicability to public
;