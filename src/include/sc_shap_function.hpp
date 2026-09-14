#pragma once

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

//! sc_shap(data, model[, name, predicted_class_only, top_k, max_attributions])
//!   -> (sample_id, class, feature_id, shap_value, base_value, sample_coverage)
//!
//! Path-dependent TreeSHAP: how much each feature pushed one sample's prediction
//! away from the model's baseline. Additive by construction --
//! `base_value + sum(shap_value)` over a sample's features equals the
//! prediction: the class probability for a classifier, the predicted value for a
//! regressor.
//!
//! A classifier explains every class, but only the PREDICTED class is returned
//! by default. With two classes the other one is an exact mirror (every value
//! negated), so nothing is lost. `predicted_class_only := false` returns them
//! all, which matters with three or more classes, where "why not the runner-up"
//! is not recoverable from the winner alone. `class` is NULL for a regressor.
//!
//! `top_k := k` keeps k features per sample (per class when all classes are
//! returned): the strongest pushes toward the prediction and the strongest
//! against it, split evenly, with the positive side taking any odd one out. If a
//! sample runs short on one side the other fills the gap, so k rows come back
//! whenever the sample has k non-zero attributions. Off by default, because
//! truncating silently hides data and breaks the additivity above.
//!
//! Features the model knows but the sample lacks are explained too: absent is
//! zero, and a zero can push a prediction as hard as a count. Features the model
//! never saw are dropped, exactly as in sc_predict -- no tree splits on them, so
//! their attribution would be 0 anyway. `sample_coverage` is the share of the
//! sample's features the model knows, as in sc_predict; at 0 the attributions
//! still add up, but they explain an all-zero row.
//!
//! Rows come out in waterfall order,
//!   ORDER BY sample_id, class, shap_value DESC
//! with ties in model column order. That is the order rows are produced in, not
//! a property DuckDB tracks: a plain SELECT or CREATE TABLE AS keeps it, joins
//! and aggregates may not, and an outer ORDER BY simply re-sorts.
//!
//! SHAP is computed densely for every sample, class and feature before any
//! `top_k` filtering, so a size guard runs first; `max_attributions` raises it.
class ScShapFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace duckdb
