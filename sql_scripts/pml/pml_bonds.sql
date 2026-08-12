create table pml_bonds
(
	ticker     text,
	name       text,
	price      double precision,
	"1d_chg"   double precision,
	"1d_pct"   double precision,
	"1y_chg"   double precision,
	"1y_pct"   double precision,
	"Z-Score"  double precision,
	"1w_pct"   double precision,
	"1w_chg"   double precision,
	"1m_chg"   double precision,
	"1m_pct"   double precision,
	"3m_chg"   double precision,
	"3m_pct"   double precision,
	"6m_chg"   double precision,
	"6m_pct"   double precision,
	mtd_chg    double precision,
	mtd_pct    double precision,
	qtd_chg    double precision,
	qtd_pct    double precision,
	ytd_chg    double precision,
	ytd_pct    double precision,
	"3Y_chg"   double precision,
	"3y_pct"   double precision,
	"5y_chg"   double precision,
	"5y_pct"   double precision,
	"10y_chg"  double precision,
	"10y_pct"  double precision,
	"52w_low"  double precision,
	"52w_high" double precision,
	country    text
		constraint pml_bonds_country_mapping_country_fk
			references country_mapping (country) not enforced,
	duration_y text
)
;

alter table pml_bonds
	owner to postgres
;