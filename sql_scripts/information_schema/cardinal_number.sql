CREATE DOMAIN information_schema.cardinal_number AS integer CONSTRAINT cardinal_number_domain_check CHECK (value >= 0);

ALTER DOMAIN information_schema.cardinal_number OWNER TO postgres;