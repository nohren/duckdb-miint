#include "minimap2_part_cursor.hpp"
#include <cassert>
#include <exception>
#include <stdexcept>

namespace miint {

Minimap2PartCursor::Minimap2PartCursor(const std::string &index_path, const Minimap2Config &config,
                                       std::function<void()> flush_freed_memory)
    : reader_(std::make_unique<Minimap2IndexReader>(index_path, config)),
      flush_freed_memory_(std::move(flush_freed_memory)) {
	current_ = reader_->ReadNextPart();
	if (!current_) {
		throw std::runtime_error("Index file '" + index_path + "' contains no parts");
	}
	// AtEof()'s probe can throw std::runtime_error too (fgetpos/fsetpos failure);
	// callers wrap everything from this constructor the same way.
	is_multi_part_ = !reader_->AtEof();
}

std::shared_ptr<SharedMinimap2Index> Minimap2PartCursor::ReleaseSinglePart() {
	assert(!is_multi_part_);
	reader_.reset();
	return std::move(current_);
}

void Minimap2PartCursor::FlushFreedMemory() {
	if (flush_freed_memory_) {
		flush_freed_memory_();
	}
}

void Minimap2PartCursor::EnsureAttached(Attachment &att, Minimap2Aligner &aligner) {
	if (att.attached && att.generation == generation_) {
		return;
	}
	// current_ is null only while a leader is between resetting it and
	// installing the next part. A thread arriving in that window (a fresh
	// worker, or one whose attachment is stale from an earlier cursor) must not
	// attach a null index; recording the current generation without attaching
	// makes its next Advance() wait on this very transition rather than spin.
	att.attached = false;
	att.generation = generation_;
	if (!current_) {
		return;
	}
	aligner.attach_shared_index(current_);
	att.attached = true;
}

bool Minimap2PartCursor::CurrentIsLastPart() {
	if (advancing_) {
		return false;
	}
	return parts_exhausted_ || !reader_ || reader_->AtEof();
}

void Minimap2PartCursor::MarkExhausted() {
	std::lock_guard<std::mutex> lock(lock_);
	parts_exhausted_ = true;
}

bool Minimap2PartCursor::Advance(Attachment &att, Minimap2Aligner &aligner, const std::function<void()> &prepare,
                                 const std::function<void()> &publish) {
	const uint64_t expected_generation = att.generation;
	// Detach FIRST, before either leading or waiting. A thread that instead
	// waited while still attached to the old part (as an earlier version of this
	// logic did) kept that part alive for the whole load — and with enough idle
	// threads sitting in that wait, for the WHOLE REST OF THE QUERY, since
	// nothing ever made them detach afterwards either. See the class comment
	// for the measurement.
	aligner.detach_shared_index();
	att.attached = false;

	// The part is held by this cursor and by every attached aligner. Threads
	// exhaust a part at different times, so whichever thread's detach above
	// happens to drop the LAST reference is the one that actually frees the
	// part's mm_idx_t — and that is not necessarily the thread that goes on to
	// become the leader below. Flush unconditionally, on every thread, right
	// after its own detach: this thread may have just triggered the real
	// deallocation even if it never becomes leader and takes the early "someone
	// else already advanced" return below.
	FlushFreedMemory();

	std::unique_lock<std::mutex> lock(lock_);

	while (true) {
		if (generation_ != expected_generation) {
			// Someone already advanced past the generation this thread was stuck on.
			return true;
		}
		if (parts_exhausted_) {
			return false;
		}
		if (advancing_) {
			cv_.wait(lock);
			continue;
		}

		// Become the leader for this transition.
		advancing_ = true;
		current_.reset(); // free the just-finished part before loading the next
		lock.unlock();

		// This reset is a second, separate potential last-reference drop (every
		// other thread may have already detached above, making the leader's own
		// reset the one that actually frees the part) — flush again here in case
		// that's what just happened.
		FlushFreedMemory();

		std::shared_ptr<SharedMinimap2Index> next_index;
		std::exception_ptr load_error;
		try {
			next_index = reader_->ReadNextPart();
			if (next_index && prepare) {
				prepare();
			}
		} catch (...) {
			load_error = std::current_exception();
		}

		lock.lock();
		advancing_ = false;
		if (load_error) {
			// Stop every other thread too — the reader is now in an unknown
			// state and cannot be trusted for a retry.
			parts_exhausted_ = true;
			cv_.notify_all();
			lock.unlock();
			std::rethrow_exception(load_error);
		}
		if (!next_index) {
			parts_exhausted_ = true;
			cv_.notify_all();
			return false;
		}

		current_ = std::move(next_index);
		if (publish) {
			publish();
		}
		generation_++;
		cv_.notify_all();
		return true;
	}
}

} // namespace miint
