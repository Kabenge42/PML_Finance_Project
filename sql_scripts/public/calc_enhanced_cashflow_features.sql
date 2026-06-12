CREATE FUNCTION public.calc_enhanced_cashflow_features(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                    text,
		        "fcf_positive_years"      integer,
		        "fcf_always_positive"     integer,
		        "capex_vs_5y_avg"         numeric,
		        "underinvestment_flag"    integer,
		        "cfo_share_of_cf"         numeric,
		        "cfi_share_of_cf"         numeric,
		        "cff_share_of_cf"         numeric,
		        "self_funding_flag"       integer,
		        "acquisition_to_fcf"      numeric,
		        "sustainable_ma_flag"     integer,
		        "fcf_4q_improvement"      numeric,
		        "cash_flow_quality_score" numeric,
		        "capex_yoy_growth"        numeric,
		        "capex_qoq_growth"        numeric,
		        "capex_3y_trend"          numeric,
		        "capex_volatility"        numeric,
		        "capex_acceleration"      integer,
		        "capex_cut_flag"          integer,
		        "overinvestment_flag"     integer,
		        "acquisitions_yoy_growth" numeric,
		        "acquisitions_vs_5y_avg"  numeric,
		        "acquisitions_ltm_total"  numeric,
		        "ma_intensity_score"      numeric,
		        "serial_acquirer_flag"    integer,
		        "acquisition_pause_flag"  integer,
		        "total_investment_to_cfo" numeric,
		        "organic_vs_inorganic"    numeric,
		        "investment_efficiency"   numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_enhanced_cashflow_features(text) OWNER TO postgres;