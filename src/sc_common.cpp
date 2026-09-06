#include "sc_common.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/parser/keyword_helper.hpp"

#include <cstring>

namespace miint {

using duckdb::InvalidInputException;

ScContext::~ScContext() {
	if (ptr) {
		sc_context_free(ptr);
	}
}

ScModel::~ScModel() {
	if (ptr) {
		sc_model_free(ptr);
	}
}

OwnedArrowArray::~OwnedArrowArray() {
	// A released array has a NULL callback, so this is a no-op on an out-slot
	// that was never filled (an sc call that failed before exporting).
	if (array_.release) {
		array_.release(&array_);
	}
	if (schema_.release) {
		schema_.release(&schema_);
	}
}

namespace {

void RequireFormat(const ArrowSchema &schema, const char *expected, const char *what) {
	if (!schema.format || std::strcmp(schema.format, expected) != 0) {
		throw InvalidInputException("%s: expected Arrow format '%s', got '%s'", what, expected,
		                            schema.format ? schema.format : "(null)");
	}
}

} // namespace

std::vector<std::string> OwnedArrowArray::ReadUtf8(const char *what) const {
	RequireFormat(schema_, "u", what);
	std::vector<std::string> out;
	if (array_.length == 0) {
		return out;
	}
	const auto *offsets = static_cast<const int32_t *>(array_.buffers[1]);
	const auto *chars = static_cast<const char *>(array_.buffers[2]);
	out.reserve(static_cast<size_t>(array_.length));
	for (int64_t i = 0; i < array_.length; i++) {
		const auto start = offsets[i];
		const auto end = offsets[i + 1];
		// An all-empty-string column allocates no data buffer at all, so guard
		// the pointer rather than the length.
		out.emplace_back(chars ? chars + start : "", static_cast<size_t>(end - start));
	}
	return out;
}

std::vector<double> OwnedArrowArray::ReadFloat64(const char *what) const {
	RequireFormat(schema_, "g", what);
	if (array_.length == 0) {
		return {};
	}
	const auto *values = static_cast<const double *>(array_.buffers[1]);
	return std::vector<double>(values, values + array_.length);
}

void ThrowSc(const char *what, sc_context_t *ctx, sc_status_t status) {
	const char *msg = ctx ? sc_context_last_error(ctx) : nullptr;
	throw InvalidInputException("%s: sc error %d%s%s", what, static_cast<int>(status), msg ? ": " : "",
	                            msg ? msg : "");
}

void LoadModelFromRelation(duckdb::Connection &conn, const std::string &relation, const char *caller,
                           sc_context_t *ctx, ScModel &out) {
	const auto q = duckdb::KeywordHelper::WriteOptionallyQuoted(relation);
	auto result = conn.Query("SELECT model FROM " + q);
	if (result->HasError()) {
		throw InvalidInputException("%s: model relation '%s' must expose a 'model' column: %s", caller, relation,
		                            result->GetError());
	}

	duckdb::Value blob;
	int64_t rows = 0;
	while (auto chunk = result->Fetch()) {
		for (duckdb::idx_t row = 0; row < chunk->size(); row++) {
			if (rows++ == 0) {
				blob = chunk->data[0].GetValue(row);
			}
		}
	}
	// A model table holds exactly one row. More than one is ambiguous -- there
	// is no basis for choosing -- and zero usually means an upstream filter ate
	// it, which is worth saying rather than failing later inside sc.
	if (rows == 0) {
		throw InvalidInputException("%s: model relation '%s' is empty", caller, relation);
	}
	if (rows > 1) {
		throw InvalidInputException("%s: model relation '%s' has %lld rows; expected exactly one", caller, relation,
		                            (long long)rows);
	}
	if (blob.IsNull()) {
		throw InvalidInputException("%s: model relation '%s' has a NULL model", caller, relation);
	}

	const auto bytes = blob.GetValueUnsafe<duckdb::string_t>();
	if (auto st = sc_model_deserialize(ctx, reinterpret_cast<const uint8_t *>(bytes.GetData()), bytes.GetSize(),
	                                   &out.ptr);
	    st != SC_OK) {
		ThrowSc(caller, ctx, st);
	}
}

} // namespace miint
