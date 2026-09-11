#include "sc_fit_function.hpp"

#include "catalog_utils.hpp"
#include "sc_coo_builder.hpp"
#include "sc_common.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/parser/keyword_helper.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "sc.h"

namespace duckdb {

namespace {

//! The metadata column holding the sample id. Everything else is a candidate
//! target.
constexpr const char *kSampleIdColumn = "sample_id";

struct ScFitData : public TableFunctionData {
	string data_relation;
	string metadata_relation;
	//! Resolved at bind time: the caller's `target_column :=`, or the metadata
	//! relation's single non-sample_id column.
	string target_column;
	//! Required. A model is selected by name, so an unnamed one is reachable
	//! only by position or by whatever hyperparameters happen to differ.
	string name;
	//! The algorithm, echoed into the output so a row says what produced it.
	string model = "random_forest";
	bool classification = false;
	int32_t n_threads = 0;
	// Defaults are sklearn's RandomForest{Classifier,Regressor} defaults, so an
	// unparameterised call matches what a user would get from scikit-learn.
	// `criterion` and `max_features` differ by task, so they are resolved in
	// Bind once `classification` is known.
	sc_rf_params_t params {};
};

//! Export one Arrow array of targets. sc takes Utf8 labels for a classifier and
//! Float64 for a regressor; the two are otherwise identical at this boundary.
struct TargetArray {
	ArrowArray array {};
	ArrowSchema schema {};
	std::vector<std::string> labels; // classification
	std::vector<double> numbers;     // regression
	std::vector<int32_t> offsets;
	std::vector<char> chars;
	const void *buffers[3] = {nullptr, nullptr, nullptr};
};

void ReleaseNothing(ArrowArray *array) {
	array->release = nullptr;
}
void ReleaseNothingSchema(ArrowSchema *schema) {
	schema->release = nullptr;
}

//! Fill `t` as an Arrow array sc can borrow. `t` owns the buffers and must
//! outlive the sc call -- sc's import borrows in place and never releases.
void BuildTargets(TargetArray &t, bool classification) {
	t.schema.format = classification ? "u" : "g";
	t.schema.name = nullptr;
	t.schema.metadata = nullptr;
	t.schema.flags = 0;
	t.schema.n_children = 0;
	t.schema.children = nullptr;
	t.schema.dictionary = nullptr;
	t.schema.release = ReleaseNothingSchema;
	t.schema.private_data = nullptr;

	int64_t length = 0;
	int64_t n_buffers = 0;
	if (classification) {
		length = static_cast<int64_t>(t.labels.size());
		t.offsets.reserve(t.labels.size() + 1);
		int32_t cursor = 0;
		t.offsets.push_back(cursor);
		for (const auto &s : t.labels) {
			t.chars.insert(t.chars.end(), s.begin(), s.end());
			cursor += static_cast<int32_t>(s.size());
			t.offsets.push_back(cursor);
		}
		t.buffers[1] = t.offsets.data();
		t.buffers[2] = t.chars.empty() ? nullptr : t.chars.data();
		n_buffers = 3;
	} else {
		length = static_cast<int64_t>(t.numbers.size());
		t.buffers[1] = t.numbers.data();
		n_buffers = 2;
	}
	// buffers[0] stays NULL: no nulls, which sc requires (null_count must be 0).
	t.array.length = length;
	t.array.null_count = 0;
	t.array.offset = 0;
	t.array.n_buffers = n_buffers;
	t.array.n_children = 0;
	t.array.buffers = t.buffers;
	t.array.children = nullptr;
	t.array.dictionary = nullptr;
	t.array.release = ReleaseNothing;
	t.array.private_data = nullptr;
}

// ---------------------------------------------------------------------------
// Input scanning
// ---------------------------------------------------------------------------

//! Scan the data relation into `builder`, rejecting NULLs.
void ScanCounts(Connection &conn, const ScFitData &bind, miint::ScCooBuilder &builder) {
	const auto q = KeywordHelper::WriteOptionallyQuoted(bind.data_relation);
	// The casts are load-bearing, not cosmetic. Reading a vector's buffer
	// directly assumes its physical type, and DuckDB infers `42.0` as
	// DECIMAL(3,1) (physical INT16), an INTEGER count column as INT32, and
	// woltka's feature ids as BIGINT or UUID. Vector::GetValue used to convert
	// on the way out; a raw buffer read cannot. Casting in SQL makes DuckDB do
	// the conversion and guarantees the layout this loop reads.
	
	//given a table with triplet columns as input to the duck db table function sc_fit_*, we are invoking the duckdb SQL engine to go fetch the data in columnar format, n output chunks of <= 2048 rows each
	auto result = conn.Query("SELECT sample_id::VARCHAR, feature_id::VARCHAR, value::DOUBLE FROM " + q);
	if (result->HasError()) {
		throw InvalidInputException(
		    "sc_fit: Data relation '%s' does not match the required COO triplet schema.\n"
		    "  Expected columns : sample_id, feature_id, value\n"
		    "  Engine error     : %s\n\n"
		    "Remedy:\n"
		    "  If your relation uses different column names (e.g. ASV/taxa names, read counts),\n"
		    "  wrap it in an aliased view before fitting:\n"
		    "    CREATE VIEW my_counts AS\n"
		    "      SELECT your_sample_col  AS sample_id,\n"
		    "             your_feature_col AS feature_id,\n"
		    "             your_count_col   AS value\n"
		    "      FROM %s;",
		    bind.data_relation, result->GetError(), bind.data_relation);
	}
	// Read through UnifiedVectorFormat rather than Vector::GetValue(row).
	// GetValue materialises a duckdb::Value per cell -- a heap-allocating
	// variant -- and ToString() allocates again on top, so a scan of n cells
	// costs ~5n allocations that are discarded immediately. At 13M cells that
	// dominates the scan. This path reads string_t views straight out of
	// DuckDB's own buffers, and ScCooBuilder::Append takes string_view, so a
	// cell now costs no allocation at all beyond interning a genuinely new id.
	//
	// Unified (not FlatVector) because the column may arrive constant- or
	// dictionary-encoded, in which case the selection vector maps row -> slot.
	// chunk is the <= 2048 record /rows coming from DuckDB in one go
	while (duckdb::unique_ptr<duckdb::DataChunk> chunk = result->Fetch()) {
		const idx_t n = chunk->size();
		// alocate a unified vector format for each column on the stack, and fill it with the data from the corresponding chunk buffer.  This is a view into the chunk's data, not a copy.
		UnifiedVectorFormat sf, ff, vf;
		chunk->data[0].ToUnifiedFormat(n, sf);
		chunk->data[1].ToUnifiedFormat(n, ff);
		chunk->data[2].ToUnifiedFormat(n, vf);
		const auto *samples = UnifiedVectorFormat::GetData<string_t>(sf);
		const auto *features = UnifiedVectorFormat::GetData<string_t>(ff);
		const auto *values = UnifiedVectorFormat::GetData<double>(vf);

		for (idx_t row = 0; row < n; row++) {
			// Step 1: Translate logical row -> physical buffer index
			const auto si = sf.sel->get_index(row);
			const auto fi = ff.sel->get_index(row);
			const auto vi = vf.sel->get_index(row);
			// A NULL here means a broken join upstream, not a zero count. An
			// absent cell is already zero in a sparse matrix, so a NULL cannot
			// be passed through and guessing at it would hide the mistake.
			if (!sf.validity.RowIsValid(si) || !ff.validity.RowIsValid(fi) || !vf.validity.RowIsValid(vi)) {
				throw InvalidInputException(
				    "sc_fit: NULL in data relation '%s' (sample_id/feature_id/value must all be non-NULL)",
				    bind.data_relation);
			}
			// Step 2: Use si to fetch the actual data from the raw buffer
			// assign the result to a readonly alias 
			// bracket operator is dereference operator
			// prevent stack copy when dereferencing the pointer via & aliasing
			const auto& s = samples[si];
			const auto& f = features[fi];
			const auto& v = values[vi];
			builder.Append(std::string_view(s.GetData(), s.GetSize()),
			               std::string_view(f.GetData(), f.GetSize()), v);
		}
	}
}

//! Scan the metadata relation into a sample_id -> target map, rejecting NULLs
//! and duplicate labels.
template <class T>
std::unordered_map<std::string, T> ScanTargets(Connection &conn, const ScFitData &bind, const char *cast) {
	const auto q = KeywordHelper::WriteOptionallyQuoted(bind.metadata_relation);
	const auto col = KeywordHelper::WriteOptionallyQuoted(bind.target_column);
	auto result = conn.Query("SELECT sample_id, " + col + "::" + cast + " FROM " + q);
	if (result->HasError()) {
		throw InvalidInputException(
		    "sc_fit: metadata relation '%s' must expose (sample_id, %s) castable to %s: %s", bind.metadata_relation,
		    bind.target_column, cast, result->GetError());
	}
	std::unordered_map<std::string, T> targets;
	while (auto chunk = result->Fetch()) {
		for (idx_t row = 0; row < chunk->size(); row++) {
			auto s = chunk->data[0].GetValue(row);
			auto t = chunk->data[1].GetValue(row);
			if (s.IsNull() || t.IsNull()) {
				throw InvalidInputException("sc_fit: NULL in metadata relation '%s' (sample_id and %s must be "
				                            "non-NULL)",
				                            bind.metadata_relation, bind.target_column);
			}
			const auto sample = s.ToString();
			// Two labels for one sample cannot be reconciled -- unlike duplicate
			// counts, there is no defensible way to combine them.
			if (!targets.emplace(sample, t.GetValue<T>()).second) {
				throw InvalidInputException(
				    "sc_fit: sample '%s' has more than one %s in metadata relation '%s'; deduplicate it first",
				    sample, bind.target_column, bind.metadata_relation);
			}
		}
	}
	return targets;
}

//! Both relations must describe exactly the same samples. Report both
//! directions at once so one round trip fixes the whole mismatch.
template <class T>
void RequireSameSamples(const std::vector<std::string> &data_samples,
                        const std::unordered_map<std::string, T> &targets, const ScFitData &bind) {
	//unlabelled: Sample is in Data, but missing from Metadata
	std::vector<std::string> unlabelled;
	for (const auto &s : data_samples) {
		if (targets.find(s) == targets.end()) {
			unlabelled.push_back(s);
		}
	}
	//undated = un-data'd: Sample is in Metadata, but missing from Data
	// in other words, which targets were missing from the target mapping above
	std::vector<std::string> undated;
	if (targets.size() + unlabelled.size() != data_samples.size()) {
		std::unordered_map<std::string, bool> present;
		for (const auto &s : data_samples) {
			present.emplace(s, true);
		}
		for (const auto &kv : targets) {
			if (present.find(kv.first) == present.end()) {
				undated.push_back(kv.first);
			}
		}
	}
	if (unlabelled.empty() && undated.empty()) {
		return;
	}

	auto sample_list = [](const std::vector<std::string> &v) {
		string out;
		const size_t shown = v.size() < 5 ? v.size() : 5;
		for (size_t i = 0; i < shown; i++) {
			out += (i ? ", " : "") + v[i];
		}
		if (v.size() > shown) {
			out += StringUtil::Format(", ... (+%llu more)", (unsigned long long)(v.size() - shown));
		}
		return out;
	};
	string msg = "sc_fit: The data and metadata relations describe different sample sets.\n"
	             "  Every sample with counts must have exactly one label, and vice versa.";
	if (!unlabelled.empty()) {
		msg += StringUtil::Format("\n  %llu sample(s) in '%s' with no %s: %s",
		                          (unsigned long long)unlabelled.size(), bind.data_relation.c_str(),
		                          bind.target_column.c_str(), sample_list(unlabelled).c_str());
	}
	if (!undated.empty()) {
		msg += StringUtil::Format("\n  %llu sample(s) in '%s' with no data: %s", (unsigned long long)undated.size(),
		                          bind.metadata_relation.c_str(), sample_list(undated).c_str());
	}
	msg += StringUtil::Format(
	    "\n\nRemedy:\n"
	    "  Restrict both sides to the samples they share, then refit:\n"
	    "    CREATE VIEW shared AS SELECT DISTINCT sample_id FROM %s SEMI JOIN %s USING (sample_id);\n"
	    "    CREATE VIEW counts AS SELECT * FROM %s SEMI JOIN shared USING (sample_id);\n"
	    "    CREATE VIEW labels AS SELECT * FROM %s SEMI JOIN shared USING (sample_id);\n"
	    "  Or fix the upstream join if the mismatch is unexpected -- a truncated metadata\n"
	    "  export trains a model on fewer samples than you think.",
	    bind.data_relation.c_str(), bind.metadata_relation.c_str(), bind.data_relation.c_str(),
	    bind.metadata_relation.c_str());
	throw InvalidInputException(msg);
}

//! Reject duplicate cells, showing the values so the caller can tell a join
//! fanout (identical values, deduplicate) from repeat measurements (differing
//! values, maybe sum). Summing a fanout silently inflates every count, so the
//! message deliberately does not prescribe a repair.
void RequireNoDuplicateCells(const miint::ScCooBuilder &builder, const ScFitData &bind) {
	const auto report = builder.FindDuplicateCells();
	if (report.Empty()) {
		return;
	}
	string msg = StringUtil::Format("sc_fit: %llu duplicate (sample_id, feature_id) cell(s) in '%s'.",
	                                (unsigned long long)report.duplicate_cells, bind.data_relation.c_str());
	for (const auto &c : report.examples) {
		string values;
		for (size_t i = 0; i < c.values.size(); i++) {
			values += (i ? ", " : "") + StringUtil::Format("%g", c.values[i]);
		}
		msg += StringUtil::Format("\n  %s / %s  x%llu  values: %s", c.sample_id.c_str(), c.feature_id.c_str(),
		                          (unsigned long long)c.count, values.c_str());
	}
	msg += "\nIdentical values suggest a join fanout (deduplicate); differing values suggest repeat "
	       "measurements (aggregate deliberately).";
	throw InvalidInputException(msg);
}

//! Fill `params` with sklearn's defaults for the task.
//!
//! Classifier: gini, max_features='sqrt'. Regressor: squared_error,
//! max_features=1.0 (all). Both: 100 trees, unbounded depth, min_samples_split=2,
//! min_samples_leaf=1, bootstrap=True, max_samples=None.
void ApplyDefaults(sc_rf_params_t &params, bool classification) {
	params.criterion = classification ? SC_CRITERION_GINI : SC_CRITERION_SQUARED_ERROR;
	params.max_depth = 0; // sklearn's None
	params.max_features.kind = classification ? SC_MAX_FEATURES_SQRT : SC_MAX_FEATURES_ALL;
	params.min_samples_split.kind = SC_MIN_SAMPLES_COUNT;
	params.min_samples_split.count = 2;
	params.min_samples_leaf.kind = SC_MIN_SAMPLES_COUNT;
	params.min_samples_leaf.count = 1;
	params.min_weight_fraction_leaf = 0.0;
	params.min_impurity_decrease = 0.0;
	params.n_estimators = 100;
	params.bootstrap = true;
	params.max_samples.kind = SC_MAX_SAMPLES_ALL;
	params.random_state = 0;
	params.optimize_feature_selection = false;
	params.rfe_step = 0.0;
	params.parameter_tuning = false;
	params.cv = 0;
	params.n_threads = 0;
}

//! Is this value an SQL integer rather than a float?
//!
//! sklearn's int/float distinction is load-bearing for the tagged unions --
//! `max_features=10` means ten features, `max_features=0.3` means thirty
//! percent -- and SQL's literal types carry exactly that information, so the
//! named parameters take ANY and dispatch on what the user actually wrote.
bool IsIntegerValue(const Value &v) {
	switch (v.type().id()) {
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
		return true;
	default:
		return false;
	}
}

void ParseMaxFeatures(const Value &v, sc_max_features_t &out) {
	if (v.type().id() == LogicalTypeId::VARCHAR) {
		const auto s = v.GetValue<string>();
		if (StringUtil::CIEquals(s, "sqrt")) {
			out.kind = SC_MAX_FEATURES_SQRT;
		} else if (StringUtil::CIEquals(s, "log2")) {
			out.kind = SC_MAX_FEATURES_LOG2;
		} else if (StringUtil::CIEquals(s, "all") || StringUtil::CIEquals(s, "none")) {
			out.kind = SC_MAX_FEATURES_ALL;
		} else {
			throw InvalidInputException("sc_fit: max_features must be 'sqrt', 'log2', 'all', a fraction in (0, 1], "
			                            "or a positive integer (got '%s')",
			                            s);
		}
		return;
	}
	if (IsIntegerValue(v)) {
		const auto n = v.GetValue<int64_t>();
		if (n <= 0) {
			throw InvalidInputException("sc_fit: max_features count must be > 0 (got %lld)", (long long)n);
		}
		out.kind = SC_MAX_FEATURES_COUNT;
		out.count = static_cast<uint64_t>(n);
		return;
	}
	const auto f = v.GetValue<double>();
	if (!(f > 0.0 && f <= 1.0)) {
		throw InvalidInputException("sc_fit: max_features fraction must be in (0, 1] (got %g)", f);
	}
	out.kind = SC_MAX_FEATURES_FRACTION;
	out.fraction = f;
}

void ParseMinSamples(const Value &v, const char *name, uint64_t min_count, sc_min_samples_t &out) {
	if (IsIntegerValue(v)) {
		const auto n = v.GetValue<int64_t>();
		if (n < static_cast<int64_t>(min_count)) {
			throw InvalidInputException("sc_fit: %s count must be >= %llu (got %lld)", name,
			                            (unsigned long long)min_count, (long long)n);
		}
		out.kind = SC_MIN_SAMPLES_COUNT;
		out.count = static_cast<uint64_t>(n);
		return;
	}
	const auto f = v.GetValue<double>();
	if (!(f > 0.0 && f <= 1.0)) {
		throw InvalidInputException("sc_fit: %s fraction must be in (0, 1] (got %g)", name, f);
	}
	out.kind = SC_MIN_SAMPLES_FRACTION;
	out.fraction = f;
}

void ParseMaxSamples(const Value &v, sc_max_samples_t &out) {
	if (v.type().id() == LogicalTypeId::VARCHAR) {
		const auto s = v.GetValue<string>();
		if (!StringUtil::CIEquals(s, "all") && !StringUtil::CIEquals(s, "none")) {
			throw InvalidInputException("sc_fit: max_samples must be 'all', a fraction in (0, 1], or a positive "
			                            "integer (got '%s')",
			                            s);
		}
		out.kind = SC_MAX_SAMPLES_ALL;
		return;
	}
	if (IsIntegerValue(v)) {
		const auto n = v.GetValue<int64_t>();
		if (n <= 0) {
			throw InvalidInputException("sc_fit: max_samples count must be > 0 (got %lld)", (long long)n);
		}
		out.kind = SC_MAX_SAMPLES_COUNT;
		out.count = static_cast<uint64_t>(n);
		return;
	}
	const auto f = v.GetValue<double>();
	if (!(f > 0.0 && f <= 1.0)) {
		throw InvalidInputException("sc_fit: max_samples fraction must be in (0, 1] (got %g)", f);
	}
	out.kind = SC_MAX_SAMPLES_FRACTION;
	out.fraction = f;
}

void ParseCriterion(const Value &v, bool classification, sc_criterion_t &out) {
	const auto s = v.GetValue<string>();
	// sc rejects a criterion that does not match the task, but catching it here
	// names both the criterion and the function the user called.
	if (classification) {
		if (StringUtil::CIEquals(s, "gini")) {
			out = SC_CRITERION_GINI;
		} else if (StringUtil::CIEquals(s, "entropy")) {
			out = SC_CRITERION_ENTROPY;
		} else {
			throw InvalidInputException("sc_fit_classifier: criterion must be 'gini' or 'entropy' (got '%s')", s);
		}
		return;
	}
	if (StringUtil::CIEquals(s, "squared_error")) {
		out = SC_CRITERION_SQUARED_ERROR;
		return;
	}
	throw InvalidInputException("sc_fit_regressor: criterion must be 'squared_error' (got '%s')", s);
}

// ---------------------------------------------------------------------------
// Bind / Execute
// ---------------------------------------------------------------------------

//! Work out which metadata column holds the target.
//!
//! With one candidate there is nothing to choose, so choose it -- `(sample_id,
//! month)` needs no configuration. With several the answer is genuinely unknown,
//! and guessing would train on the wrong variable without any error: a
//! `(sample_id, age, bmi, delivery_mode)` table has three equally plausible
//! targets and picking one silently would be the worst outcome.
//!
//! This replaces a hardcoded default of "value", which happened to suit
//! long-format metadata and quietly failed for everything else.
std::string ResolveTargetColumn(Connection &conn, const std::string &relation) {
	const auto q = KeywordHelper::WriteOptionallyQuoted(relation);
	// LIMIT 0 binds the relation and returns its schema without scanning it.
	auto probe = conn.Query("SELECT * FROM " + q + " LIMIT 0");
	if (probe->HasError()) {
		throw InvalidInputException("sc_fit: metadata relation '%s' could not be read: %s", relation,
		                            probe->GetError());
	}

	std::vector<std::string> candidates;
	bool has_sample_id = false;
	for (const auto &name : probe->names) {
		if (StringUtil::CIEquals(name, kSampleIdColumn)) {
			has_sample_id = true;
			continue; // if sample_id, dont push to candidates
		}
		candidates.push_back(name);
	}
	if (!has_sample_id) {
		throw InvalidInputException("sc_fit: metadata relation '%s' has no 'sample_id' column", relation);
	}
	if (candidates.empty()) {
		throw InvalidInputException(
		    "sc_fit: metadata relation '%s' has only a sample_id column; it needs a target column too", relation);
	}
	if (candidates.size() == 1) {
		return candidates[0];
	}

	std::string list;
	for (size_t i = 0; i < candidates.size(); i++) {
		list += (i ? ", " : "") + ("'" + candidates[i] + "'");
	}
	throw InvalidInputException(
	    "sc_fit: Metadata relation '%s' has more than one column that could be the target.\n"
	    "  Candidates : [%s]\n\n"
	    "Remedy:\n"
	    "  Name the variable you are modelling, so the wrong one cannot be picked silently:\n"
	    "    SELECT * FROM sc_fit_regressor('counts', '%s',\n"
	    "                                   target_column := 'one_of_the_above',\n"
	    "                                   name := 'my_model');\n"
	    "  A metadata relation with exactly one non-sample_id column needs no target_column at all.",
	    relation, list, relation);
}

unique_ptr<FunctionData> ScFitBind(ClientContext &context, TableFunctionBindInput &input,
                                   vector<LogicalType> &return_types, vector<string> &names, bool classification) {
	auto data = make_uniq<ScFitData>();
	data->classification = classification;
	data->data_relation = input.inputs[0].GetValue<string>();
	data->metadata_relation = input.inputs[1].GetValue<string>();
	if (data->data_relation.empty() || data->metadata_relation.empty()) {
		throw InvalidInputException("sc_fit: data and metadata relation names must not be empty");
	}

	ApplyDefaults(data->params, classification);
	auto &params = data->params;
	for (auto &kv : input.named_parameters) {
		const auto &k = kv.first;
		const auto &v = kv.second;
		if (v.IsNull()) {
			throw InvalidInputException("sc_fit: named parameter '%s' must not be NULL", k);
		}
		if (StringUtil::CIEquals(k, "name")) {
			data->name = v.GetValue<string>();
			if (data->name.empty()) {
				throw InvalidInputException("sc_fit: name must not be empty; omit it to leave the model unnamed");
			}
		} else if (StringUtil::CIEquals(k, "target_column")) {
			data->target_column = v.GetValue<string>();
		} else if (StringUtil::CIEquals(k, "n_threads")) {
			data->n_threads = v.GetValue<int32_t>();
			params.n_threads = data->n_threads;
		} else if (StringUtil::CIEquals(k, "model")) {
			const auto m = v.GetValue<string>();
			if (!StringUtil::CIEquals(m, "random_forest")) {
				throw InvalidInputException("sc_fit: model must be 'random_forest' (got '%s')", m);
			}
			data->model = "random_forest"; // normalised, so the column is stable
		} else if (StringUtil::CIEquals(k, "n_estimators")) {
			params.n_estimators = v.GetValue<int64_t>();
			if (params.n_estimators <= 0) {
				throw InvalidInputException("sc_fit: n_estimators must be > 0 (got %lld)",
				                            (long long)params.n_estimators);
			}
		} else if (StringUtil::CIEquals(k, "random_state")) {
			params.random_state = static_cast<uint64_t>(v.GetValue<int64_t>());
		} else if (StringUtil::CIEquals(k, "max_depth")) {
			// <= 0 is sklearn's None (unbounded), which sc reads the same way.
			params.max_depth = v.GetValue<int64_t>();
		} else if (StringUtil::CIEquals(k, "criterion")) {
			ParseCriterion(v, classification, params.criterion);
		} else if (StringUtil::CIEquals(k, "max_features")) {
			ParseMaxFeatures(v, params.max_features);
		} else if (StringUtil::CIEquals(k, "min_samples_split")) {
			ParseMinSamples(v, "min_samples_split", 2, params.min_samples_split);
		} else if (StringUtil::CIEquals(k, "min_samples_leaf")) {
			ParseMinSamples(v, "min_samples_leaf", 1, params.min_samples_leaf);
		} else if (StringUtil::CIEquals(k, "min_weight_fraction_leaf")) {
			params.min_weight_fraction_leaf = v.GetValue<double>();
		} else if (StringUtil::CIEquals(k, "min_impurity_decrease")) {
			params.min_impurity_decrease = v.GetValue<double>();
		} else if (StringUtil::CIEquals(k, "bootstrap")) {
			params.bootstrap = v.GetValue<bool>();
		} else if (StringUtil::CIEquals(k, "max_samples")) {
			ParseMaxSamples(v, params.max_samples);
		}
	}
	// Only probe when the caller did not say. An explicit target_column
	// short-circuits this entirely.
	if (data->target_column.empty()) {
		auto conn = MakeReadOnlyHelperConnection(context);
		data->target_column = ResolveTargetColumn(conn, data->metadata_relation);
	}

	// Required rather than optional. A name is not an input to the fit -- the
	// same seed with and without one produces a byte-identical model -- but
	// making it mandatory removes a state from the system: `name` is never NULL,
	// so `sc_predict(..., name := ...)` always works and nobody has to reason
	// about the unnamed case. Optional names get skipped, and then a registry
	// needs ALTER TABLE and rowid archaeology to become selectable.
	if (data->name.empty()) {
		throw InvalidInputException(
		    "sc_fit: name is required -- `name := 'my_model'`. It is how a model is selected later, e.g. "
		    "sc_predict(data, models, name := 'my_model'). Any label will do if you are just exploring.");
	}

	// sklearn rejects max_samples without bootstrap rather than silently ignoring
	// it; sc has no opinion, so catch it here where the message can be specific.
	if (!params.bootstrap && params.max_samples.kind != SC_MAX_SAMPLES_ALL) {
		throw InvalidInputException("sc_fit: max_samples is only meaningful with bootstrap := true");
	}

	// `name` is required, so this column is never NULL and a registry is
	// selectable from the moment it is created. Nothing is ever invented to fill
	// it -- a generated 'm0' would collide on the very next call, since a table
	// function cannot see what already exists in the table its row is headed for.
	//
	// `random_state` is echoed because it is the one thing needed to reproduce a
	// fit that the caller would otherwise have to remember and re-type. Richer
	// provenance stays the caller's: `SELECT 'x' AS target, * FROM sc_fit_...`.
	// `model` names the algorithm and mirrors the `model :=` parameter; the
	// serialized forest lives in `model_blob`. They were one column called
	// `model`, which meant `model := 'random_forest'` went in and bytes came
	// out under the same name -- and left a row unable to say which algorithm
	// produced it once there is more than one.
	names = {"name", "model", "model_blob", "n_samples", "n_features", "n_trees", "task", "random_state"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BLOB,    LogicalType::BIGINT,
	                LogicalType::BIGINT,  LogicalType::BIGINT,  LogicalType::VARCHAR, LogicalType::BIGINT};
	return std::move(data);
}

struct ScFitGlobalState : public GlobalTableFunctionState {
	bool done = false;
};

unique_ptr<GlobalTableFunctionState> ScFitInitGlobal(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<ScFitGlobalState>();
}

/*
	Entry point called by DuckDB execution engine to pull output chunks.
	receives query context, function input state wrappers, and destination chunk
	the output we write to. In this case a single row is produced, that is the fitted model.
*/
void ScFitExecute(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
	// global state to tell us if we are finished
	auto& gstate = input.global_state->Cast<ScFitGlobalState>();
	if (gstate.done) {
		output.SetCardinality(0); // turn off the engine
		return;
	}
	// write to first byte in gstate (global duck db state) so that next time we come through here we know we are done
	//output.SetCardinality(0); is a contract for table functions to tell the engine that we are done producing output, so we don't produce any more rows
	// executes ScFitExecute exactly once
	gstate.done = true;

	const auto &bind = input.bind_data->Cast<ScFitData>();
	auto conn = MakeReadOnlyHelperConnection(context);

	// instantiate the builder class on the function stack, intake triplets, canoncialise them, and produce a sparse matrix.
	miint::ScCooBuilder builder;
	ScanCounts(conn, bind, builder);
	if (builder.NumNonZeros() == 0) {
		throw InvalidInputException("sc_fit: data relation '%s' produced no cells", bind.data_relation);
	}
	// remove duplicate triplets using sorted flat array of bitwise packed values
	// prefetchable, cache-friendly and less memory than a hash table. The builder's internal state is now a non duplicated sparse matrix in COO format, ready to be packaged to arrow and passed to sc
	RequireNoDuplicateCells(builder, bind);

	// Targets are positional: element i must be the label for sample index i.
	// The order comes from the builder's sorted dictionary, looked up through a
	// hash -- never from a second ORDER BY, whose collation need not match
	// std::string's byte ordering.
	TargetArray targets;
	std::vector<std::string> data_samples;
	auto table = std::unique_ptr<miint::ScCooTable> {};

	if (bind.classification) {
		auto labels = ScanTargets<string>(conn, bind, "VARCHAR");
		table = builder.Finalize();
		//sorted sample ids after Finalize()
		data_samples = table->SampleIds();
		RequireSameSamples(data_samples, labels, bind);
		targets.labels.reserve(data_samples.size());
		for (const auto &s : data_samples) {
			targets.labels.push_back(labels.at(s));
		}
	} else {
		auto values = ScanTargets<double>(conn, bind, "DOUBLE");
		table = builder.Finalize();
		//sorted sample ids after Finalize()
		data_samples = table->SampleIds();
		RequireSameSamples(data_samples, values, bind);
		targets.numbers.reserve(data_samples.size());
		for (const auto &s : data_samples) {
			targets.numbers.push_back(values.at(s));
		}
	}
	BuildTargets(targets, bind.classification);

	// Fully resolved at bind time, so a bad parameter fails the query before any
	// scanning happens.
	const sc_rf_params_t &params = bind.params;

	//multi-threaded execution context for sc, with the number of threads specified by the user. If n_threads is 0, sc will use all available threads.
	sc_config_t config {};
	config.n_threads = bind.n_threads;
	miint::ScContext ctx;
	if (auto st = sc_context_new(&config, &ctx.ptr); st != SC_OK) {
		miint::ThrowSc("sc_fit", nullptr, st);
	}

	// hand over the arrow data to sc, which will take ownership of the buffers and free them when done. The table is now owned by sc and must not be freed by the caller.
	// Fit the model and serialize it into a blob
	miint::ScModel model;
	const auto fit = bind.classification ? sc_fit_classifier : sc_fit_regressor;
	if (auto st = fit(ctx.ptr, table->get(), &targets.array, &targets.schema, &params, &model.ptr); st != SC_OK) {
		miint::ThrowSc(bind.classification ? "sc_fit_classifier" : "sc_fit_regressor", ctx.ptr, st);
	}

	uint8_t* blob = nullptr;
	size_t blob_len = 0;
	// hand over **blob, one more level of indirection, to copy over the value of the memory addr mapped to the variable blob and capture that inside the scope of the C function so it can dereference it and write to it.
	if (auto st = sc_model_serialize(model.ptr, &blob, &blob_len); st != SC_OK) {
		miint::ThrowSc("sc_model_serialize", ctx.ptr, st);
	}
	// sc owns this buffer until sc_buffer_free; copy it into DuckDB's heap first. Then free it from sc's heap. This is a one-time copy, so the model blob is now owned by DuckDB.
	duckdb::Value blob_value = Value::BLOB(blob, blob_len);
	sc_buffer_free(blob, blob_len);

	//DuckDB's table function output is a single row with the fitted model, so we set the cardinality to 1 and fill in the columns with the model's metadata and serialized blob.
	output.SetCardinality(1); // one row returned, which is the fitted model
	// The output columns are:
	output.SetValue(0, 0, Value(bind.name));
	output.SetValue(1, 0, Value(bind.model));
	output.SetValue(2, 0, blob_value);
	output.SetValue(3, 0, Value::BIGINT(table->NumSamples()));
	output.SetValue(4, 0, Value::BIGINT(table->NumFeatures()));
	output.SetValue(5, 0, Value::BIGINT(params.n_estimators));
	output.SetValue(6, 0, Value(bind.classification ? "classification" : "regression"));
	output.SetValue(7, 0, Value::BIGINT(static_cast<int64_t>(params.random_state)));
}
/**
 * Creates a new table function for fitting a model based on sc rf
 */
TableFunction MakeFitFunction(const char *name, table_function_bind_t bind) {
	// declare a variable named fn and initialize an instance of the TableFunction class with the constructor call. The TableFunction constructor takes the following parameters:
	TableFunction fn(name, {LogicalType::VARCHAR, LogicalType::VARCHAR}, ScFitExecute, bind, ScFitInitGlobal);
	fn.named_parameters["name"] = LogicalType::VARCHAR;
	fn.named_parameters["target_column"] = LogicalType::VARCHAR;
	fn.named_parameters["n_estimators"] = LogicalType::BIGINT;
	fn.named_parameters["random_state"] = LogicalType::BIGINT;
	fn.named_parameters["n_threads"] = LogicalType::INTEGER;
	fn.named_parameters["model"] = LogicalType::VARCHAR;
	fn.named_parameters["max_depth"] = LogicalType::BIGINT;
	fn.named_parameters["criterion"] = LogicalType::VARCHAR;
	fn.named_parameters["min_weight_fraction_leaf"] = LogicalType::DOUBLE;
	fn.named_parameters["min_impurity_decrease"] = LogicalType::DOUBLE;
	fn.named_parameters["bootstrap"] = LogicalType::BOOLEAN;
	// ANY, not a fixed type: these are sklearn's tagged unions, where an integer
	// means a count and a float means a fraction. SQL literal types carry that
	// distinction already, so ParseMaxFeatures / ParseMinSamples / ParseMaxSamples
	// dispatch on what the user wrote.
	fn.named_parameters["max_features"] = LogicalType::ANY;
	fn.named_parameters["min_samples_split"] = LogicalType::ANY;
	fn.named_parameters["min_samples_leaf"] = LogicalType::ANY;
	fn.named_parameters["max_samples"] = LogicalType::ANY;
	return fn;
}

unique_ptr<FunctionData> ScFitClassifierBind(ClientContext &context, TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types, vector<string> &names) {
	return ScFitBind(context, input, return_types, names, /*classification=*/true);
}
unique_ptr<FunctionData> ScFitRegressorBind(ClientContext &context, TableFunctionBindInput &input,
                                            vector<LogicalType> &return_types, vector<string> &names) {
	return ScFitBind(context, input, return_types, names, /*classification=*/false);
}

} // namespace

void ScFitFunction::Register(ExtensionLoader &loader) {
	loader.RegisterFunction(MakeFitFunction("sc_fit_classifier", ScFitClassifierBind));
	loader.RegisterFunction(MakeFitFunction("sc_fit_regressor", ScFitRegressorBind));
}

} // namespace duckdb
