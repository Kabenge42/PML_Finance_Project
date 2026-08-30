CREATE TABLE analytics.accounting_anomaly_analysis
(
	ticker                   text,
	accounting_anomaly_score double precision
);

ALTER TABLE analytics.accounting_anomaly_analysis
	OWNER TO postgres;