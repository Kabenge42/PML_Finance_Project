CREATE TABLE bonds
(
	name               text,
	currency           text,
	issuer             text,
	isin               text,
	sector             text,
	maturity           double precision,
	"S&P Rating"       text,
	coupon             double precision,
	"Coupon frequency" text,
	"YTM (Ask)"        double precision,
	ask                double precision,
	"Minimum size"     double precision,
	"Country of issue" text,
	"Amount issued"    double precision,
	"Increment size"   double precision,
	subordinated       text,
	"EU Tax"           text,
	"W/Tax"            text,
	"Accrued days"     double precision,
	"Accrued interest" double precision
);

ALTER TABLE bonds
	OWNER TO postgres;