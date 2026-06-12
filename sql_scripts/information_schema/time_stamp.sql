CREATE DOMAIN information_schema.time_stamp AS timestamp(2) with time zone DEFAULT CURRENT_TIMESTAMP(2);

ALTER DOMAIN information_schema.time_stamp OWNER TO postgres;