create table information_schema.sql_implementation_info
(
    implementation_info_id   information_schema.character_data,
    implementation_info_name information_schema.character_data,
    integer_value            information_schema.cardinal_number,
    character_value          information_schema.character_data,
    comments                 information_schema.character_data
);

alter table information_schema.sql_implementation_info
    owner to postgres;

grant select on information_schema.sql_implementation_info to public;

