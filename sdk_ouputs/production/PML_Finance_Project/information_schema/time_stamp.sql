create domain information_schema.time_stamp as timestamp(2) with time zone
    default CURRENT_TIMESTAMP(2);

alter domain information_schema.time_stamp owner to postgres;

