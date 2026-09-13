#pragma once

#include "duckdb/main/connection.hpp"

#include <string>
#include <vector>

#include "sc.h"

namespace miint {

//! An sc context, freed on destruction.
//!
//! sc's handles are opaque heap allocations with their own destructors. Wrapping
//! them keeps the many error paths in the table functions from leaking on a
//! throw.
struct ScContext {
	sc_context_t *ptr = nullptr; // nullptr so other programs know it is uninitialized and should not be dereferenced
	ScContext() = default; // default constructor for the struct
	~ScContext(); // destructor
	ScContext(const ScContext &) = delete; // forbid copy constructor, since we don't want to copy the pointer. Preventing double free bugs.
	ScContext &operator=(const ScContext &) = delete; // forbid copy assignment operator, since we don't want to copy the pointer. Preventing double free bugs.
};

//! A trained sc model, freed on destruction.
struct ScModel {
	sc_model_t *ptr = nullptr;
	ScModel() = default;
	~ScModel();
	ScModel(const ScModel &) = delete;
	ScModel &operator=(const ScModel &) = delete;
};

//! An Arrow array sc exported to us, released on destruction.
//!
//! Note the ownership direction, which is the reverse of `ScCooBuilder`'s. sc's
//! *imports* borrow -- it reads our buffers in place and never releases them --
//! but its *exports* move: `to_ffi` installs a release callback and the consumer
//! must invoke it exactly once. Skip it and the buffers leak; call it twice and
//! it is a double free.
class OwnedArrowArray {
public:
	OwnedArrowArray() = default;
	~OwnedArrowArray();
	OwnedArrowArray(const OwnedArrowArray &) = delete;
	OwnedArrowArray &operator=(const OwnedArrowArray &) = delete;

	//! Out-slots for an sc `export_*` call.
	ArrowArray *array() {
		return &array_;
	}
	ArrowSchema *schema() {
		return &schema_;
	}

	//! The Arrow format string sc stamped on the export, or nullptr if the
	//! out-slot was never filled. "u" = Utf8, "g" = Float64.
	const char *Format() const {
		return schema_.format;
	}

	//! Decode a `Utf8` array (format "u"): buffers are [validity, offsets, data]
	//! and row i is data[offsets[i] .. offsets[i + 1]].
	std::vector<std::string> ReadUtf8(const char *what) const;
	//! Decode a `Float64` array (format "g"): buffers are [validity, data].
	std::vector<double> ReadFloat64(const char *what) const;

	//! Decode a `FixedSizeList<Float64>[width]` (format "+w:N") as one flat
	//! row-major vector of `length * width` doubles, returning `width`.
	//!
	//! This is the first NESTED layout we read. The outer array carries no
	//! values of its own -- `n_buffers` is 1 (validity only) and the doubles
	//! live in `children[0]`, a plain Float64 array of `length * width`. The
	//! width is not a field: it is encoded in the format string itself, after
	//! the "+w:" prefix.
	//!
	//! Kept flat rather than nested because every caller immediately walks it
	//! row by row -- proba is one row per sample of n_classes, SHAP one row per
	//! sample of n_outputs * n_features.
	std::vector<double> ReadFixedSizeListFloat64(const char *what, int64_t &width) const;

private:
	ArrowArray array_ {};
	ArrowSchema schema_ {};
};

//! Turn a non-zero `sc_status_t` into a DuckDB error, preferring sc's own
//! message from the context's last-error slot.
[[noreturn]] void ThrowSc(const char *what, sc_context_t *ctx, sc_status_t status);

//! Read the `task` column of a model relation: "classification" or
//! "regression".
//!
//! A table function must declare its return types at bind time, and a
//! prediction column is VARCHAR for a classifier and DOUBLE for a regressor.
//! The model already knows which it is -- sc's `sc_predict` dispatches on
//! `forest.task()` -- but sc exposes no C accessor for it, so the fit writes it
//! into the model relation and this reads it back. Deciding the schema from the
//! data is the same thing `read_csv` does when it sniffs a file.
std::string ReadModelTask(duckdb::Connection &conn, const std::string &relation, const std::string &name,
                          const char *caller);

//! Build the `WHERE name = '...'` clause for a model lookup, or an empty string
//! when no name was given.
//!
//! Only the *value* comes from the caller, and it goes through
//! `KeywordHelper::WriteQuoted`. That is what makes this safe where a general
//! `where :=` parameter would not be: the predicate's shape is ours, so there is
//! no way to inject a second clause.
std::string ModelNameFilter(const std::string &name);

//! Read a serialized model out of `relation`'s `model_blob` column and
//! deserialize it.
//!
//! With `name` empty the relation itself must hold exactly one row. With a name,
//! rows are filtered to that name first -- so one table can be a registry of
//! many models and a call still names exactly one, without the caller having to
//! create a view to narrow it.
void LoadModelFromRelation(duckdb::Connection &conn, const std::string &relation, const std::string &name,
                           const char *caller, sc_context_t *ctx, ScModel &out);

} // namespace miint
