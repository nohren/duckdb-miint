#include "catch2/catch_all.hpp"

#include "sc_coo_builder.hpp"

#include <string>
#include <vector>

using miint::ScCooBuilder;
using miint::ScCooTable;

namespace {

// Read an Arrow Int64 array's data buffer. buffers[1] is the data buffer --
// buffers[0] is validity, which these arrays never have.
std::vector<int64_t> ReadInt64(const ArrowArray &a) {
	const auto *p = static_cast<const int64_t *>(a.buffers[1]);
	return std::vector<int64_t>(p, p + a.length);
}

std::vector<double> ReadFloat64(const ArrowArray &a) {
	const auto *p = static_cast<const double *>(a.buffers[1]);
	return std::vector<double>(p, p + a.length);
}

// Utf8 is [validity, offsets, data]; row i is data[offsets[i] .. offsets[i+1]].
std::vector<std::string> ReadUtf8(const ArrowArray &a) {
	const auto *offsets = static_cast<const int32_t *>(a.buffers[1]);
	const auto *chars = static_cast<const char *>(a.buffers[2]);
	std::vector<std::string> out;
	for (int64_t i = 0; i < a.length; i++) {
		out.emplace_back(chars + offsets[i], static_cast<size_t>(offsets[i + 1] - offsets[i]));
	}
	return out;
}

// The exact contents of data/biom/test.biom as read_biom emits them.
struct Cell {
	const char *sample;
	const char *feature;
	double value;
};

// Deliberately NOT in sorted order: the builder must normalise it.
const std::vector<Cell> BIOM_CELLS = {
    {"Sample6", "GG_OTU_4", 1.0}, {"Sample3", "GG_OTU_1", 1.0}, {"Sample1", "GG_OTU_2", 5.0},
    {"Sample4", "GG_OTU_3", 4.0}, {"Sample2", "GG_OTU_5", 1.0}, {"Sample6", "GG_OTU_2", 1.0},
    {"Sample3", "GG_OTU_4", 1.0}, {"Sample5", "GG_OTU_2", 3.0}, {"Sample1", "GG_OTU_4", 2.0},
    {"Sample2", "GG_OTU_2", 1.0}, {"Sample3", "GG_OTU_5", 1.0}, {"Sample4", "GG_OTU_2", 2.0},
    {"Sample2", "GG_OTU_4", 1.0}, {"Sample6", "GG_OTU_3", 2.0}, {"Sample3", "GG_OTU_3", 1.0},
};

} // namespace

TEST_CASE("ScCooBuilder encodes a real BIOM table", "[sc_coo]") {
	ScCooBuilder builder;
	for (const auto &c : BIOM_CELLS) {
		builder.Append(c.sample, c.feature, c.value);
	}
	auto table = builder.Finalize();
	REQUIRE(table);

	// data/biom/test.biom is 6 samples x 5 features with 15 stored cells.
	REQUIRE(table->NumSamples() == 6);
	REQUIRE(table->NumFeatures() == 5);
	REQUIRE(table->NumNonZeros() == 15);

	// Both dictionaries come out sorted even though the input was shuffled.
	// This is what makes the integer encoding reproducible: a model trained on
	// one query's row order must stay valid for another's.
	REQUIRE(table->SampleIds() ==
	        std::vector<std::string> {"Sample1", "Sample2", "Sample3", "Sample4", "Sample5", "Sample6"});
	REQUIRE(table->FeatureIds() ==
	        std::vector<std::string> {"GG_OTU_1", "GG_OTU_2", "GG_OTU_3", "GG_OTU_4", "GG_OTU_5"});
}

TEST_CASE("ScCooBuilder indices point at the sorted dictionaries", "[sc_coo]") {
	ScCooBuilder builder;
	for (const auto &c : BIOM_CELLS) {
		builder.Append(c.sample, c.feature, c.value);
	}
	auto table = builder.Finalize();
	REQUIRE(table);

	const auto rows = ReadInt64(table->get()->rows);
	const auto cols = ReadInt64(table->get()->cols);
	const auto vals = ReadFloat64(table->get()->vals);
	REQUIRE(rows.size() == BIOM_CELLS.size());

	// Every triple must still name the cell it was appended with, once decoded
	// through the sorted dictionaries. This is the test that would fail if the
	// remap after sorting were wrong -- the failure mode that silently trains a
	// model on transposed features.
	for (size_t i = 0; i < BIOM_CELLS.size(); i++) {
		INFO("cell " << i);
		CHECK(table->SampleIds()[static_cast<size_t>(rows[i])] == BIOM_CELLS[i].sample);
		CHECK(table->FeatureIds()[static_cast<size_t>(cols[i])] == BIOM_CELLS[i].feature);
		CHECK(vals[i] == BIOM_CELLS[i].value);
	}

	// Indices stay in range -- sc's from_coo rejects anything outside.
	for (auto r : rows) {
		CHECK((r >= 0 && r < table->NumSamples()));
	}
	for (auto c : cols) {
		CHECK((c >= 0 && c < table->NumFeatures()));
	}
}

TEST_CASE("ScCooBuilder emits arrays sc will accept", "[sc_coo]") {
	ScCooBuilder builder;
	builder.Append("s1", "f1", 1.5);
	auto table = builder.Finalize();
	REQUIRE(table);
	const auto *t = table->get();

	// sc-arrow's check_primitive rejects null_count != 0 and offset != 0, and
	// matches the format string exactly.
	struct Expect {
		const ArrowArray *array;
		const ArrowSchema *schema;
		const char *format;
		int64_t n_buffers;
	};
	const std::vector<Expect> expected = {
	    {&t->rows, &t->rows_schema, "l", 2},
	    {&t->cols, &t->cols_schema, "l", 2},
	    {&t->vals, &t->vals_schema, "g", 2},
	    {&t->sample_ids, &t->sample_ids_schema, "u", 3},
	    {&t->feature_ids, &t->feature_ids_schema, "u", 3},
	};
	for (const auto &e : expected) {
		INFO("format " << e.format);
		CHECK(std::string(e.schema->format) == e.format);
		CHECK(e.array->null_count == 0);
		CHECK(e.array->offset == 0);
		CHECK(e.array->n_buffers == e.n_buffers);
		CHECK(e.array->n_children == 0);
		CHECK(e.array->release != nullptr);
		// No nulls anywhere, so the validity buffer is absent -- a NULL
		// pointer, not a bitmap of ones.
		CHECK(e.array->buffers[0] == nullptr);
	}
}

TEST_CASE("ScCooBuilder keeps duplicate cells for sc to sum", "[sc_coo]") {
	// sc-core's from_coo sums duplicate (row, col) pairs (scipy COO->CSR
	// semantics). Collapsing them here would be a silent behaviour change, so
	// the builder must pass all three through.
	ScCooBuilder builder;
	builder.Append("s1", "f1", 2.0);
	builder.Append("s1", "f1", 3.0);
	builder.Append("s1", "f1", 5.0);
	auto table = builder.Finalize();
	REQUIRE(table);

	CHECK(table->NumNonZeros() == 3);
	CHECK(table->NumSamples() == 1);
	CHECK(table->NumFeatures() == 1);
	CHECK(ReadFloat64(table->get()->vals) == std::vector<double> {2.0, 3.0, 5.0});
}

TEST_CASE("ScCooBuilder offsets count bytes, not characters", "[sc_coo]") {
	// Arrow Utf8 offsets are byte counts. A multi-byte id would corrupt every
	// later id if they were treated as character counts.
	ScCooBuilder builder;
	builder.Append("sample", "ae", 1.0);
	builder.Append("sample", "\xc3\xa9", 2.0); // "e" with acute: 1 char, 2 bytes
	auto table = builder.Finalize();
	REQUIRE(table);

	const auto &ids = table->get()->feature_ids;
	const auto *offsets = static_cast<const int32_t *>(ids.buffers[1]);
	REQUIRE(ids.length == 2);
	CHECK(offsets[0] == 0);
	// Sorted: "ae" (2 bytes) then "\xc3\xa9" (2 bytes).
	CHECK(offsets[1] == 2);
	CHECK(offsets[2] == 4);
	CHECK(ReadUtf8(ids) == std::vector<std::string> {"ae", "\xc3\xa9"});
}

TEST_CASE("ScCooBuilder rejects an empty table", "[sc_coo]") {
	// sc's from_coo errors on a 0-row or 0-column matrix, so there is no valid
	// table to hand back. Fail here rather than build one sc will refuse.
	ScCooBuilder builder;
	CHECK(builder.Finalize() == nullptr);
}

TEST_CASE("ScCooBuilder is reusable after Finalize", "[sc_coo]") {
	ScCooBuilder builder;
	builder.Append("s1", "f1", 1.0);
	auto first = builder.Finalize();
	REQUIRE(first);
	CHECK(builder.NumNonZeros() == 0);

	// The second table must not inherit the first's dictionaries.
	builder.Append("other", "feat", 9.0);
	auto second = builder.Finalize();
	REQUIRE(second);
	CHECK(second->NumSamples() == 1);
	CHECK(second->SampleIds() == std::vector<std::string> {"other"});
	// The first table's buffers stay valid and untouched.
	CHECK(first->SampleIds() == std::vector<std::string> {"s1"});
	CHECK(ReadFloat64(first->get()->vals) == std::vector<double> {1.0});
}

TEST_CASE("ScCooBuilder handles an empty-string id", "[sc_coo]") {
	// An empty id is a real value, distinct from absent. It sorts first and
	// contributes a zero-width span to the offsets buffer.
	ScCooBuilder builder;
	builder.Append("s1", "zzz", 1.0);
	builder.Append("s1", "", 2.0);
	auto table = builder.Finalize();
	REQUIRE(table);

	REQUIRE(table->NumFeatures() == 2);
	CHECK(table->FeatureIds() == std::vector<std::string> {"", "zzz"});
	const auto cols = ReadInt64(table->get()->cols);
	CHECK(cols[0] == 1); // "zzz"
	CHECK(cols[1] == 0); // ""
}

TEST_CASE("ScCooBuilder reports duplicate cells with their values", "[sc_coo]") {
	// A join fanout: the same cell appended twice with identical values. The
	// repair is to deduplicate; summing would inflate the count. Differing
	// values would mean genuine repeat measurements, where summing is right.
	// The builder cannot tell them apart, so it reports both keys AND values
	// and leaves the choice to the caller.
	ScCooBuilder builder;
	builder.Append("Sample1", "GG_OTU_2", 5.0);
	builder.Append("Sample2", "GG_OTU_2", 1.0);
	builder.Append("Sample2", "GG_OTU_2", 1.0); // fanout: identical
	builder.Append("Sample3", "GG_OTU_4", 2.0);
	builder.Append("Sample3", "GG_OTU_4", 7.0); // repeat measurement: differs

	const auto report = builder.FindDuplicateCells();
	REQUIRE_FALSE(report.Empty());
	CHECK(report.duplicate_cells == 2);
	REQUIRE(report.examples.size() == 2);

	// Examples arrive in packed (row, col) order, which follows insertion order
	// here since Sample1/2/3 intern as 0/1/2.
	CHECK(report.examples[0].sample_id == "Sample2");
	CHECK(report.examples[0].feature_id == "GG_OTU_2");
	CHECK(report.examples[0].count == 2);
	CHECK(report.examples[0].values == std::vector<double> {1.0, 1.0});

	CHECK(report.examples[1].sample_id == "Sample3");
	CHECK(report.examples[1].values == std::vector<double> {2.0, 7.0});
}

TEST_CASE("ScCooBuilder finds no duplicates in clean data", "[sc_coo]") {
	ScCooBuilder builder;
	for (const auto &c : BIOM_CELLS) {
		builder.Append(c.sample, c.feature, c.value);
	}
	// data/biom/test.biom is a matrix: one cell per (sample, feature).
	CHECK(builder.FindDuplicateCells().Empty());

	// Same feature across different samples is not a duplicate, nor is the same
	// sample across different features -- only the pair counts.
	ScCooBuilder b2;
	b2.Append("s1", "f1", 1.0);
	b2.Append("s1", "f2", 1.0);
	b2.Append("s2", "f1", 1.0);
	CHECK(b2.FindDuplicateCells().Empty());
}

TEST_CASE("ScCooBuilder caps the duplicate examples it collects", "[sc_coo]") {
	// The count must be complete even though the examples are bounded -- an
	// error message says "N duplicates" and shows a handful.
	ScCooBuilder builder;
	for (int i = 0; i < 20; i++) {
		const auto f = "f" + std::to_string(i);
		builder.Append("s1", f, 1.0);
		builder.Append("s1", f, 1.0);
	}
	const auto report = builder.FindDuplicateCells(3);
	CHECK(report.duplicate_cells == 20);
	CHECK(report.examples.size() == 3);
}

TEST_CASE("ScCooBuilder duplicate scan handles trivial inputs", "[sc_coo]") {
	ScCooBuilder empty;
	CHECK(empty.FindDuplicateCells().Empty());

	ScCooBuilder single;
	single.Append("s1", "f1", 1.0);
	CHECK(single.FindDuplicateCells().Empty());
}
