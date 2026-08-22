create table pml.currency_mapping
(
	unit      text
		constraint currency_mapping_pk
			unique,
	unit_name text
)
;

alter table pml.currency_mapping
	owner to postgres
;