CREATE TABLE pml.screening_global_yields
(
	ticker     text,
	name       text,
	price      double precision,
	"1D Chg"   double precision,
	"1D %"     double precision,
	"1Y Chg"   double precision,
	"1Y %"     double precision,
	"Z-Score"  double precision,
	"1W %"     double precision,
	"1W Chg"   double precision,
	"1M Chg"   double precision,
	"1M %"     double precision,
	"3M Chg"   double precision,
	"3M %"     double precision,
	"6M Chg"   double precision,
	"6M %"     double precision,
	"MTD Chg"  double precision,
	"MTD %"    double precision,
	"QTD Chg"  double precision,
	"QTD %"    double precision,
	"YTD Chg"  double precision,
	"YTD %"    double precision,
	"3Y Chg"   double precision,
	"3Y %"     double precision,
	"5Y Chg"   double precision,
	"5Y %"     double precision,
	"10Y Chg"  double precision,
	"10Y %"    double precision,
	"52W Low"  double precision,
	"52W High" double precision
);

ALTER TABLE pml.screening_global_yields
	OWNER TO postgres;