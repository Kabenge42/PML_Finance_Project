CREATE VIEW information_schema.collation_character_set_applicability
			(collation_catalog, collation_schema, collation_name, character_set_catalog, character_set_schema,
			 character_set_name) AS
-- missing source code
;

ALTER TABLE information_schema.collation_character_set_applicability
	OWNER TO postgres;

GRANT SELECT ON information_schema.collation_character_set_applicability TO PUBLIC;