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

//! One `(sample_id, feature_id)` pair that was appended more than once, with the
//! values that were seen for it.
//!
//! The values matter as much as the keys: identical values point at a join
//! fanout (a duplicated key upstream, where one row is spurious), differing
//! values at genuine repeat measurements. The correct repair is opposite in the
//! two cases -- deduplicating versus summing -- and summing a fanout silently
//! inflates every affected count, so a caller reporting this should show them.
struct DuplicateCell {
	std::string sample_id;
	std::string feature_id;
	size_t count = 0;
	std::vector<double> values;
};

//! What [`ScCooBuilder::FindDuplicateCells`] found.
struct DuplicateReport {
	//! Distinct `(sample_id, feature_id)` pairs appearing more than once.
	size_t duplicate_cells = 0;
	//! A bounded sample of them, for an error message.
	std::vector<DuplicateCell> examples;

	bool Empty() const {
		return duplicate_cells == 0;
	}
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

	//! Report `(sample_id, feature_id)` pairs appended more than once.
	//!
	//! Policy-free by design: sc's `from_coo` *sums* duplicates (scipy COO
	//! semantics), which is right for genuine repeat measurements and wrong for
	//! a join fanout. The builder cannot tell those apart, so it reports and
	//! lets the caller decide.
	//!
	//! Costs one temporary `uint64` per stored cell (8 bytes, freed on return)
	//! plus an O(n log n) sort -- deliberately not a hash set, whose per-node
	//! overhead would be several times larger. That matters for the wasm build,
	//! where DuckDB and this extension share one linear memory.
	//!
	//! Safe to call before [`Finalize`]; it reads the interned indices and does
	//! not modify them.
	DuplicateReport FindDuplicateCells(size_t max_examples = 5) const;

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
