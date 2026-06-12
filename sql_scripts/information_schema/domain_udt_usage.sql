CREATE VIEW information_schema.domain_udt_usage
			(udt_catalog, udt_schema, udt_name, domain_catalog, domain_schema, domain_name) AS
-- missing source code
;

ALTER TABLE information_schema.domain_udt_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.domain_udt_usage TO PUBLIC;