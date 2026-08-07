CREATE FUNCTION information_schema._pg_index_position(oid, smallint) RETURNS integer
	STABLE STRICT
	LANGUAGE sql
BEGIN ATOMIC
 SELECT (ss.a).n AS n
    FROM ( SELECT information_schema._pg_expandarray(pg_index.indkey) AS a
            FROM pg_index
           WHERE (pg_index.indexrelid = $1)) ss
   WHERE ((ss.a).x = $2);
END;

ALTER FUNCTION information_schema._pg_index_position(unknown, unknown) OWNER TO postgres;