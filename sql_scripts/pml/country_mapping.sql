create table pml.country_mapping
(
	country      text
		constraint country__pk
			unique
		constraint country_mapping_currency_mapping_unit_fk
			references pml.currency_mapping (unit) not enforced,
	country_name text
)
;

alter table pml.country_mapping
	owner to postgres
;