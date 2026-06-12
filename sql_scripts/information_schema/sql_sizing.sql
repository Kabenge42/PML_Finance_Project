CREATE TABLE information_schema.sql_sizing
(
	sizing_id       information_schema.cardinal_number,
	sizing_name     information_schema.character_data,
	supported_value information_schema.cardinal_number,
	comments        information_schema.character_data
);

ALTER TABLE information_schema.sql_sizing
	OWNER TO postgres;

GRANT SELECT ON information_schema.sql_sizing TO PUBLIC;