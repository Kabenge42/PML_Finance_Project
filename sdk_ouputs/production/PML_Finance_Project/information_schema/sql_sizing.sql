create table information_schema.sql_sizing
(
    sizing_id       information_schema.cardinal_number,
    sizing_name     information_schema.character_data,
    supported_value information_schema.cardinal_number,
    comments        information_schema.character_data
);

alter table information_schema.sql_sizing
    owner to postgres;

grant select on information_schema.sql_sizing to public;

