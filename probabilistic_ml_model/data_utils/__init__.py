"""
Data utilities for the probabilistic_ml_model package.

Re-exports key functions from data_utils and inference_schema submodules
so callers can import from the subpackage directly:

    from probabilistic_ml_model.data_utils import (
        load_equities_data_from_db,
        load_feature_data_from_db,
        backfill_feature_columns,
        ARVIZ_AVAILABLE,
        build_monte_carlo_inference_data,
        summarize_inference_data,
    )
"""

from probabilistic_ml_model.data_utils.data_utils import (
    aggregate_probability_results,
    load_equities_data_from_db,
    load_feature_data_from_db,
    load_all_feature_views,
    backfill_feature_columns,
    ExportConfig,
    export_to_db,
    export_to_csv,
    export_to_json,
    reorder_with_identifiers,
    load_identifier_columns,
    get_identifier_cols_set,
    get_equities_schema,
    get_view_category_mapping,
    compute_metric_statistics,
    validate_feature_alignment,
    load_feature_categories_from_db,
)

# Inference schema re-exports (guarded — ArviZ is optional)
try:
    from probabilistic_ml_model.data_utils.inference_schema import (
        ARVIZ_AVAILABLE,
        EquityCoordinates,
        IdentifierCoordinates,
        FeatureCoordinates,
        EquitiesSchemaMetadata,
        FeatureRegistryMetadata,
        FeatureViewSpec,
        EquitiesMaterializedViewSpec,
        build_monte_carlo_inference_data,
        build_beat_probability_inference_data,
        build_credit_risk_inference_data,
        build_accounting_anomaly_inference_data,
        build_category_analysis_inference_data,
        build_feature_view_inference_data,
        build_resampled_technical_inference_data,
        summarize_inference_data,
        load_identifier_coordinates_from_db,
        load_equities_schema_metadata_from_db,
        load_feature_registry_metadata_from_db,
        load_feature_view_spec_from_db,
        load_mv_equities_spec_from_db,
        FEATURE_VIEW_REGISTRY,
    )
except ImportError:
    ARVIZ_AVAILABLE = False
