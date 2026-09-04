#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "sc.h"

namespace miint {

//! Owns every buffer behind an `sc_coo_table_t` and releases them on destruction.
//!
//! sc borrows across the Arrow C Data Interface — `sc-arrow`'s `import_*` reads
//! the caller's buffers in place and never invokes `release` — so this object
//! must outlive any sc call that reads it.
class ScCooTable {
public:
	ScCooTable() = default;
	~ScCooTable();
	ScCooTable(const ScCooTable &) = delete;
	ScCooTable &operator=(const ScCooTable &) = delete;

	//! Borrowed; valid while this object lives.
	const sc_coo_table_t *get() const {
		return &table_;
	}
	int64_t NumSamples() const {
		return table_.n_samples;
	}
	int64_t NumFeatures() const {
		return table_.n_features;
	}
	//! Number of stored triples. Duplicates are NOT collapsed here — sc sums
	//! them (scipy COO semantics, sc-core `matrix.rs` `from_coo`).
	int64_t NumNonZeros() const {
		return table_.rows.length;
	}

	//! Dictionaries, in the sorted order the `cols` / `rows` indices refer to.
	const std::vector<std::string> &SampleIds() const {
		return sample_ids_;
	}
	const std::vector<std::string> &FeatureIds() const {
		return feature_ids_;
	}

private:
	friend class ScCooBuilder;
	sc_coo_table_t table_ {};
	std::vector<std::string> sample_ids_;
	std::vector<std::string> feature_ids_;
};

//! Accumulates long-format `(sample_id, feature_id, value)` cells and emits the
//! COO feature table sc expects.
//!
//! This is the bridge between the shape DuckDB has and the shape sc wants.
//! `read_biom`, `woltka_ogu_per_sample`, and anything else projected to those
//! three columns all arrive here — sc is coupled to the triple, not to a file
//! format.
//!
//! **Both dictionaries are sorted**, so a given set of ids always produces the
//! same integer encoding regardless of the order rows arrive in. That matters
//! beyond tidiness: a model trained when `OTU_7` was column 6 returns nonsense
//! if prediction data encodes it as column 2, and sc only validates the feature
//! *count* (sc-core `model.rs` `check_features`), never the identity. It also
//! matches the reference pipeline sc's own fixtures were generated with
//! (`oracle/generator/gen_forest.py` `load_aligned`, which sorts both axes).
class ScCooBuilder {
public:
	//! Ids are copied on first sight and interned thereafter.
	void Append(std::string_view sample_id, std::string_view feature_id, double value);

	//! Sorts both dictionaries, remaps the interned indices, and builds the
	//! Arrow arrays. The builder is left empty and reusable.
	//!
	//! Returns nullptr if nothing was appended: sc rejects a 0-row or 0-column
	//! matrix outright, so an empty table has no valid representation.
	std::unique_ptr<ScCooTable> Finalize();

	size_t NumNonZeros() const {
		return vals_.size();
	}
	size_t NumSamples() const {
		return sample_index_.size();
	}
	size_t NumFeatures() const {
		return feature_index_.size();
	}

private:
	//! Intern `id`, returning its provisional (insertion-order) index.
	static int64_t Intern(std::unordered_map<std::string, int64_t> &index, std::vector<std::string> &ids,
	                      std::string_view id);

	std::unordered_map<std::string, int64_t> sample_index_;
	std::unordered_map<std::string, int64_t> feature_index_;
	std::vector<std::string> sample_ids_;
	std::vector<std::string> feature_ids_;
	std::vector<int64_t> rows_;
	std::vector<int64_t> cols_;
	std::vector<double> vals_;
};

} // namespace miint
