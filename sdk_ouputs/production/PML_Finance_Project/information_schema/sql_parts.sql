create table information_schema.sql_parts
(
    feature_id     information_schema.character_data,
    feature_name   information_schema.character_data,
    is_supported   information_schema.yes_or_no,
    is_verified_by information_schema.character_data,
    comments       information_schema.character_data
);

alter table information_schema.sql_parts
    owner to postgres;

