create function information_schema._pg_char_octet_length(typid oid, typmod integer) returns integer
    immutable
    strict
    parallel safe
    language sql
RETURN CASE
           WHEN (typid = ANY (ARRAY [(25)::oid, (1042)::oid, (1043)::oid])) THEN CASE
                                                                                     WHEN (typmod = '-1'::integer)
                                                                                         THEN (((2)::double precision ^ (30)::double precision))::integer
                                                                                     ELSE (
                                                                                         _pg_char_max_length(typid, typmod) *
                                                                                         pg_encoding_max_length((SELECT pg_database.encoding
                                                                                                                 FROM pg_database
                                                                                                                 WHERE (pg_database.datname = current_database())))) END
           ELSE NULL::integer END;

alter function information_schema._pg_char_octet_length(unknown, unknown) owner to postgres;

