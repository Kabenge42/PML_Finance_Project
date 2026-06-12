CREATE VIEW information_schema.character_sets
			(character_set_catalog, character_set_schema, character_set_name, character_repertoire, form_of_use,
			 default_collate_catalog, default_collate_schema, default_collate_name)
AS
-- missing source code
;

ALTER TABLE information_schema.character_sets
	OWNER TO postgres;

GRANT SELECT ON information_schema.character_sets TO PUBLIC;