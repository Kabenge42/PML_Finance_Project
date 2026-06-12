CREATE DOMAIN information_schema.yes_or_no AS varchar(3) CONSTRAINT yes_or_no_check CHECK ((value)::text = ANY
                                                                                           ((ARRAY ['YES'::character varying, 'NO'::character varying])::text[]));

ALTER DOMAIN information_schema.yes_or_no OWNER TO postgres;