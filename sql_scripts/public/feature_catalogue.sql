DROP TABLE IF EXISTS feature_catalogue;
create table feature_catalogue
(
    category         varchar(64),
    feature_alias    varchar(128),
    source_function  varchar(128),
    calculation_type varchar(32),
    data_type        varchar(32)
);

alter table feature_catalogue
    owner to postgres;