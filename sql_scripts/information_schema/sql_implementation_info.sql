CREATE TABLE information_schema.sql_implementation_info
(
	implementation_info_id   information_schema.character_data,
	implementation_info_name information_schema.character_data,
	integer_value            information_schema.cardinal_number,
	character_value          information_schema.character_data,
	comments                 information_schema.character_data
);

ALTER TABLE information_schema.sql_implementation_info
	OWNER TO postgres;

GRANT SELECT ON information_schema.sql_implementation_info TO PUBLIC;