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
	sc_context_t *ptr = nullptr;
	ScContext() = default;
	~ScContext();
	ScContext(const ScContext &) = delete;
	ScContext &operator=(const ScContext &) = delete;
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

	//! Decode a `Utf8` array (format "u"): buffers are [validity, offsets, data]
	//! and row i is data[offsets[i] .. offsets[i + 1]].
	std::vector<std::string> ReadUtf8(const char *what) const;
	//! Decode a `Float64` array (format "g"): buffers are [validity, data].
	std::vector<double> ReadFloat64(const char *what) const;

private:
	ArrowArray array_ {};
	ArrowSchema schema_ {};
};

//! Turn a non-zero `sc_status_t` into a DuckDB error, preferring sc's own
//! message from the context's last-error slot.
[[noreturn]] void ThrowSc(const char *what, sc_context_t *ctx, sc_status_t status);

//! Read a serialized model out of `relation`'s `model` column and deserialize
//! it. The relation must hold exactly one row -- a model table produced by
//! `CREATE TABLE m AS SELECT * FROM sc_fit_*(...)`.
void LoadModelFromRelation(duckdb::Connection &conn, const std::string &relation, const char *caller,
                           sc_context_t *ctx, ScModel &out);

} // namespace miint
