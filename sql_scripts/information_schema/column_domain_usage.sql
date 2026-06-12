CREATE VIEW information_schema.column_domain_usage
			(domain_catalog, domain_schema, domain_name, table_catalog, table_schema, table_name, column_name) AS
-- missing source code
;

ALTER TABLE information_schema.column_domain_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.column_domain_usage TO PUBLIC;