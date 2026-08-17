create table currency_mapping
(
	unit      text
		constraint currency_mapping_pk
			unique,
	unit_name text
)
;

alter table currency_mapping
	owner to postgres
;