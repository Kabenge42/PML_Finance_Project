create domain information_schema.yes_or_no as varchar(3)
    constraint yes_or_no_check check ((VALUE)::text = ANY
                                      ((ARRAY ['YES'::character varying, 'NO'::character varying])::text[]));

alter domain information_schema.yes_or_no owner to postgres;

