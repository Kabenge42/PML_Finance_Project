create table analytics."09_gate_report_v2"
(
	gate         text,
	status       text,
	value        text,
	threshold    text,
	blocking     boolean,
	detail       text,
	rationale    text,
	run_id       text,
	exported_at  timestamp with time zone,
	source_sha   text,
	source_dirty boolean
)
;

alter table analytics."09_gate_report_v2"
	owner to postgres
;