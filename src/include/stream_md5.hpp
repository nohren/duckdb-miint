#pragma once

#include "duckdb/common/crypto/md5.hpp"
#include "duckdb/common/exception.hpp"
#include "read_ena_sequences_policy.hpp"
#include <string>

namespace miint {

// Incrementally hashes raw bytes as they stream past (see duckdb_seq_read's
// gzip/direct-read taps) and, once the underlying source has reached its true
// EOF, compares the accumulated digest against an ENA-reported fastq_md5.
//
// Mirrors the duckdb::MD5Context idiom used by GzipMd5Stream in
// ena_upload_reads.cpp (src/ena_upload_reads.cpp:364-368), but in the
// opposite direction: that struct hashes bytes on their way INTO a gzip
// encoder for upload; this one hashes bytes coming OUT of a download and
// compares against an externally supplied digest rather than just reporting
// the one it computed.
//
// An empty `expected_md5_hex` means ENA did not report an md5 for this file
// (or the caller chose not to verify it) -- both Add and VerifyOrThrow become
// no-ops so callers can construct a StreamMd5 unconditionally and let the
// empty case fall through cheaply rather than branching at every call site.
class StreamMd5 {
public:
	explicit StreamMd5(std::string expected_md5_hex) : expected_(std::move(expected_md5_hex)) {
	}

	void Add(duckdb::const_data_ptr_t data, duckdb::idx_t len) {
		if (expected_.empty() || len == 0) {
			return;
		}
		ctx_.Add(data, len);
	}

	// Finalizes the digest and throws duckdb::IOException on mismatch. `label`
	// identifies the run/file in the error message. No-op when the expected
	// md5 was empty, or when already finalized (safe to call from multiple
	// cleanup paths without double-throwing).
	//
	// The message is a published contract, not an incidental string: see
	// docs/insdc_ena.md ("Message contract") for what it guarantees and why,
	// and test/cpp/test_ena_md5.cpp for the pin -- rewording it is breaking.
	void VerifyOrThrow(const std::string &label) {
		if (expected_.empty() || finished_) {
			return;
		}
		finished_ = true;
		std::string actual = ctx_.FinishHex();
		if (actual != expected_) {
			throw duckdb::IOException(miint::BuildMd5MismatchMessage(label, expected_, actual));
		}
	}

private:
	std::string expected_;
	duckdb::MD5Context ctx_;
	bool finished_ = false;
};

} // namespace miint
