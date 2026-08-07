CREATE TABLE country_mapping
(
	country      text
		CONSTRAINT country__pk UNIQUE
		CONSTRAINT country_mapping_currency_mapping_unit_fk REFERENCES currency_mapping (unit) NOT ENFORCED,
	country_name text
);

ALTER TABLE country_mapping
	OWNER TO postgres;