#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>

#include "KreppIndexBuilder.hpp"

// krepp's thread-count global: defined in ext/krepp/src/common.cpp, declared in
// its common.hpp. Declared here rather than by including that header, because
// ${KREPP_DIR}/src is deliberately kept off every include path except the two
// TUs that need it - krepp ships a klib set (kseq.h, kvec.h, kalloc.h, ...)
// whose basenames collide with minimap2's and would shadow them everywhere else
// in this build. See CMakeLists.txt, set_source_files_properties for
// src/KreppPlacer.cpp. If krepp ever moves this into a namespace, this fails to
// link, which is the loud failure and not a silent one.
extern uint32_t num_threads;

namespace {

// Nothing can be read here, so BuildKreppIndex fails at the first filesystem
// access. Everything this test observes has already happened by then, which is
// the point: learning one integer should not cost a real index build.
const char *const kMissingInput = "/nonexistent/krepp-index-builder-test/input.tsv";

} // namespace

// Either the requested thread count reaches krepp or it does not, and nothing
// else in the codebase observes it. The SQL tests structurally cannot: the
// strongest thing krepp_index_create_threads.test asserts is that the index is
// byte-identical at 1 and 4 threads - which is equally true of a build that
// honours `threads` and one that ignores it entirely. Deleting set_num_threads
// from BuildKreppIndex was measured to pass all 83 assertions across the three
// krepp SQL test files. This is the check that fails when that happens.
TEST_CASE("BuildKreppIndex hands its thread count to krepp", "[krepp]") {
	if (!miint::KreppIndexThreadsSupported()) {
		// No OpenMP means krepp has no parallel regions to size, and
		// krepp_index_create refuses threads > 1 long before reaching here.
		// SKIP rather than `return`: a bare return still reports the case as
		// PASSED, so a build that silently lost OpenMP would show up only as a
		// drop in the assertion total (27 -> 24, measured) while still printing
		// "All tests passed". SKIP says so in the run summary instead.
		SKIP("krepp was built without OpenMP, so there is no thread count to set");
	}
	miint::InstallKreppErrorHandler();

	miint::KreppIndexOptions options;
	options.input_map_path = kMissingInput;
	options.index_dir = (std::filesystem::temp_directory_path() / "krepp_threads_probe").string();
	options.threads = 4;

	// Anti-vacuity: park the global on a value the assertion below rejects, so
	// a pass cannot come from stale state left by another test or an earlier
	// build in this process.
	num_threads = 1;
	REQUIRE(num_threads != options.threads);

	// set_num_threads is the first effectful statement in BuildKreppIndex, so
	// it has run by the time krepp rejects the missing input map. The throw is
	// incidental - it is only what stops the build before it does real work.
	CHECK_THROWS(miint::BuildKreppIndex(options));
	CHECK(num_threads == options.threads);
}
