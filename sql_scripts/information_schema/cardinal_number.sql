create domain information_schema.cardinal_number as integer
    constraint cardinal_number_domain_check check (VALUE >= 0);

alter domain information_schema.cardinal_number owner to postgres;

