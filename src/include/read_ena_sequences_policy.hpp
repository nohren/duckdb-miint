#pragma once

#include <cstddef>
#include <string>

namespace miint {

// Policy and message formatting for a run's integrity failure detected at
// completion (PerRunReader::Finish()): an md5 digest mismatch at true EOF, or
// a bad ascp exit code.
//
// ShouldSkipRunIntegrityFailure returns true if the run should be
// skipped-with-warning (recorded in skipped_runs, surfaced via
// miint_warnings()) rather than thrown, which would abort the whole scan.
//
// Skip only when there is a sibling run to protect. A single-run scan keeps the
// hard throw: there is nothing else to discard, and a downstream that resolves a
// scalar accession one run per call (the Qiita ENA ingest job) relies on that
// error to fail the run. A multi-run scan -- a `varchar[]` of accessions, or a
// project accession that expands to many runs -- must NOT let one corrupt run
// abort the query and throw away every sibling already downloaded, so it skips
// the bad run and lets the rest land. (By true EOF the skipped run's rows were
// already emitted downstream; the warning flags them as unverified, the same
// contract as the mid-stream-truncation skip.)
inline bool ShouldSkipRunIntegrityFailure(size_t total_runs) {
	return total_runs > 1;
}

// The mismatch message is a published contract (docs/insdc_ena.md, "Message
// contract"), pinned by test/cpp/test_ena_md5.cpp; rewording it is breaking.
inline std::string BuildMd5MismatchMessage(const std::string &label, const std::string &expected,
                                           const std::string &actual) {
	return "read_ena_sequences: md5 mismatch for '" + label + "': ENA reported " + expected +
	       " but downloaded bytes hash to " + actual;
}

} // namespace miint
