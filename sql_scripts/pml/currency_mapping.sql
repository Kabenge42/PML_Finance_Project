CREATE TABLE currency_mapping
(
	unit      text
		CONSTRAINT currency_mapping_pk UNIQUE,
	unit_name text
);

ALTER TABLE currency_mapping
	OWNER TO postgres;