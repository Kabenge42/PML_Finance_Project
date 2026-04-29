create table currencies
(
    "Date"         text,
    d0             text,
    d1             text,
    currency       text,
    unit           numeric,
    "Value"        double precision,
    reference_date date
);

alter table currencies
    owner to postgres;

