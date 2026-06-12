CREATE VIEW information_schema.collations(collation_catalog, collation_schema, collation_name, pad_attribute) AS
-- missing source code
;

ALTER TABLE information_schema.collations
	OWNER TO postgres;

GRANT SELECT ON information_schema.collations TO PUBLIC;