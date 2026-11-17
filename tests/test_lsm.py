import os
import tempfile
import unittest

from lsm import CompactionError, LSMEngine, Options
from lsm.compaction import KIND_DELETE_EMPTY, KIND_MERGE, KIND_TRIVIAL_MOVE


def small_options(**overrides):
    base = dict(
        max_levels=6,
        l0_compaction_trigger=2,
        max_file_size=120,
        level_size_base=200,
        level_size_multiplier=4,
        memtable_size_limit=10 ** 9,
    )
    base.update(overrides)
    return Options(**base)


class EngineCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name

    def open_engine(self, **overrides):
        return LSMEngine.open(self.dir, small_options(**overrides))

    def sst_files(self):
        return sorted(os.listdir(os.path.join(self.dir, "sst")))


class TestPointLookup(EngineCase):
    def test_memtable_hit_and_miss(self):
        eng = self.open_engine()
        eng.put("a", "1")
        self.assertEqual(eng.get("a"), "1")
        self.assertIsNone(eng.get("missing"))

    def test_newest_version_wins_across_levels(self):
        eng = self.open_engine()
        eng.put("k", "old")
        eng.flush()
        eng.compact_all()
        self.assertGreater(eng.version.num_non_empty_levels(), 0)
        eng.put("k", "new")
        eng.flush()
        self.assertEqual(eng.get("k"), "new")
        eng.compact_all()
        self.assertEqual(eng.get("k"), "new")

    def test_l0_visibility_order_newest_file_first(self):
        eng = self.open_engine()
        eng.put("x", "v1")
        eng.flush()
        eng.put("x", "v2")
        eng.flush()
        self.assertEqual(len(eng.version.files(0)), 2)
        self.assertEqual(eng.get("x"), "v2")

    def test_delete_hides_value(self):
        eng = self.open_engine()
        eng.put("d", "v")
        eng.flush()
        eng.delete("d")
        self.assertIsNone(eng.get("d"))
        eng.flush()
        self.assertIsNone(eng.get("d"))


class TestRangeScan(EngineCase):
    def test_scan_merges_sources_and_filters_tombstones(self):
        eng = self.open_engine()
        for key in ["a", "b", "c", "d"]:
            eng.put(key, key.upper())
        eng.flush()
        eng.put("b", "B2")
        eng.delete("c")
        eng.put("e", "E")
        self.assertEqual(
            eng.scan(),
            [("a", "A"), ("b", "B2"), ("d", "D"), ("e", "E")],
        )

    def test_scan_bounds(self):
        eng = self.open_engine()
        for i in range(10):
            eng.put(f"k{i}", str(i))
        eng.flush()
        self.assertEqual(eng.scan("k2", "k4"), [("k2", "2"), ("k3", "3"), ("k4", "4")])

    def test_scan_after_compaction(self):
        eng = self.open_engine()
        for i in range(20):
            eng.put(f"key{i:02d}", f"v{i}")
        eng.flush()
        eng.delete("key05")
        eng.flush()
        eng.compact_all()
        keys = [k for k, _ in eng.scan()]
        self.assertEqual(len(keys), 19)
        self.assertNotIn("key05", keys)


class TestDeleteThenReinsert(EngineCase):
    def test_reinsert_after_delete_survives_compaction(self):
        eng = self.open_engine()
        eng.put("k", "v1")
        eng.flush()
        eng.delete("k")
        eng.flush()
        eng.put("k", "v2")
        eng.flush()
        self.assertEqual(eng.get("k"), "v2")
        eng.compact_all()
        self.assertEqual(eng.get("k"), "v2")
        self.assertEqual(eng.scan(), [("k", "v2")])

    def test_delete_after_reinsert(self):
        eng = self.open_engine()
        eng.put("k", "v1")
        eng.delete("k")
        eng.put("k", "v2")
        eng.delete("k")
        eng.flush()
        self.assertIsNone(eng.get("k"))
        eng.compact_all()
        self.assertIsNone(eng.get("k"))


class TestTombstoneRules(EngineCase):
    def test_tombstone_kept_while_deeper_level_may_hold_value(self):
        eng = self.open_engine()
        eng.inject_file(3, {"k": (1, "old-value")})
        eng.inject_file(2, {"k": (2, "older-value"), "p": (3, "x")})
        pads = {f"pad{i}": (10 + i, "y" * 30) for i in range(8)}
        pads["k"] = (5, None)
        eng.inject_file(1, pads)

        task = eng.pick_compaction()
        self.assertIsNotNone(task)
        self.assertEqual(task.kind, KIND_MERGE)
        self.assertEqual(task.output_level, 2)
        handle = eng.run_compaction(task)
        out_keys = {}
        for meta in handle.outputs:
            out_keys.update(eng._entries_of(meta.file_id))
        self.assertIn("k", out_keys)
        self.assertIsNone(out_keys["k"][1], "tombstone must survive: L3 may hold older value")
        self.assertIsNone(eng.get("k"))

        eng.compact_all()
        self.assertIsNone(eng.get("k"))
        for meta in eng.version.all_files():
            self.assertNotIn("k", eng._entries_of(meta.file_id))

    def test_tombstone_dropped_at_bottommost_level(self):
        eng = self.open_engine()
        eng.put("gone", "v")
        eng.flush()
        eng.delete("gone")
        eng.flush()
        eng.compact_all()
        for meta in eng.version.all_files():
            self.assertNotIn("gone", eng._entries_of(meta.file_id))
        self.assertGreaterEqual(eng.stats["tombstones_dropped"], 1)

    def test_old_versions_collapsed(self):
        eng = self.open_engine()
        for i in range(5):
            eng.put("k", f"v{i}")
            eng.flush()
        eng.compact_all()
        seen = []
        for meta in eng.version.all_files():
            entries = eng._entries_of(meta.file_id)
            if "k" in entries:
                seen.append(entries["k"])
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][1], "v4")


class TestOverlappingRanges(EngineCase):
    def test_l0_compaction_pulls_in_all_overlapping_files(self):
        eng = self.open_engine()
        eng.put("a", "1")
        eng.put("m", "1")
        eng.flush()
        eng.put("f", "2")
        eng.put("t", "2")
        eng.flush()
        eng.put("b", "3")
        eng.flush()
        task = eng.pick_compaction()
        self.assertIsNotNone(task)
        self.assertEqual(task.input_level, 0)
        self.assertEqual(len(task.inputs), 3, "overlapping L0 ranges must compact together")

    def test_grandparents_included_in_merge(self):
        eng = self.open_engine()
        eng.inject_file(1, {"a": (4, "x"), "b": (5, "y")})
        eng.inject_file(2, {"b": (2, "old-b"), "c": (3, "z")})
        eng.put("a", "new-a")
        eng.flush()
        eng.compact_all()
        self.assertEqual(eng.get("a"), "new-a")
        self.assertEqual(eng.get("b"), "y")
        self.assertEqual(eng.get("c"), "z")


class TestEmptyFiles(EngineCase):
    def test_empty_file_scheduled_for_deletion(self):
        eng = self.open_engine()
        meta = eng.inject_file(2, {})
        self.assertTrue(meta.empty)
        self.assertIsNone(eng.get("anything"))
        task = eng.pick_compaction()
        self.assertIsNotNone(task)
        self.assertEqual(task.kind, KIND_DELETE_EMPTY)
        eng.run_compaction(task)
        self.assertIsNone(eng.version.find(meta.file_id))

    def test_merge_producing_no_output_deletes_inputs(self):
        eng = self.open_engine()
        eng.delete("only-tombstone")
        eng.flush()
        eng.compact_all()
        self.assertEqual(eng.version.all_files(), [])
        self.assertIsNone(eng.get("only-tombstone"))


class TestInProgressWrites(EngineCase):
    def test_picker_skips_files_of_running_compaction(self):
        eng = self.open_engine(l0_compaction_trigger=2)
        for i in range(4):
            eng.put(f"k{i}", "v")
            eng.flush()
        task = eng.pick_compaction()
        handle = eng.begin_compaction(task)
        busy = task.all_input_ids()
        self.assertTrue(busy)
        again = eng.pick_compaction()
        if again is not None:
            self.assertFalse(again.all_input_ids() & busy)
        eng.abort_compaction(handle)

    def test_begin_conflicting_compaction_rejected(self):
        eng = self.open_engine()
        for i in range(3):
            eng.put(f"k{i}", "v")
            eng.flush()
        task = eng.pick_compaction()
        eng.begin_compaction(task)
        with self.assertRaises(CompactionError):
            eng.begin_compaction(task)

    def test_flush_allowed_while_compaction_in_progress(self):
        eng = self.open_engine()
        for i in range(3):
            eng.put(f"k{i}", "v")
            eng.flush()
        task = eng.pick_compaction()
        handle = eng.begin_compaction(task)
        eng.put("new", "data")
        eng.flush()
        self.assertEqual(eng.get("new"), "data")
        eng.abort_compaction(handle)


class TestCompactionFailure(EngineCase):
    def test_failure_rolls_back_partial_outputs(self):
        eng = self.open_engine(max_file_size=60)
        for i in range(6):
            eng.put(f"key{i % 3}", f"value-{i}-" + "x" * 20)
            eng.flush()
        before = [f.file_id for f in eng.version.all_files()]
        before_ssts = self.sst_files()
        task = eng.pick_compaction()
        self.assertIsNotNone(task)
        with self.assertRaises(CompactionError):
            eng.run_compaction(task, fail_after_writes=1)
        after = [f.file_id for f in eng.version.all_files()]
        self.assertEqual(before, after, "failed compaction must not change the version")
        self.assertEqual(before_ssts, self.sst_files(), "partial outputs must be removed")
        for i in range(3):
            self.assertEqual(eng.get(f"key{i}"), f"value-{i + 3}-" + "x" * 20)
        handle = eng.run_compaction()
        self.assertIsNotNone(handle)


class TestCrashRecovery(EngineCase):
    def test_committed_state_survives_reopen(self):
        eng = self.open_engine()
        eng.put("a", "1")
        eng.put("b", "2")
        eng.flush()
        eng.delete("a")
        eng.put("c", "3")
        eng.close()
        del eng

        eng2 = self.open_engine()
        self.assertIsNone(eng2.get("a"))
        self.assertEqual(eng2.get("b"), "2")
        self.assertEqual(eng2.get("c"), "3")

    def test_uncommitted_compaction_outputs_are_orphaned(self):
        eng = self.open_engine()
        for i in range(3):
            eng.put("k", f"v{i}")
            eng.flush()
        baseline = [f.file_id for f in eng.version.all_files()]
        eng.run_compaction(commit=False)
        orphan_count = len(self.sst_files())
        self.assertGreater(orphan_count, len(baseline))
        del eng

        eng2 = self.open_engine()
        self.assertEqual(baseline, [f.file_id for f in eng2.version.all_files()])
        self.assertEqual(len(self.sst_files()), len(baseline))
        self.assertEqual(eng2.get("k"), "v2")

    def test_wal_replay_after_crash_without_flush(self):
        eng = self.open_engine()
        eng.put("pending", "value")
        eng.put("gone", "x")
        eng.delete("gone")
        del eng

        eng2 = self.open_engine()
        self.assertEqual(eng2.get("pending"), "value")
        self.assertIsNone(eng2.get("gone"))


class TestLevelChanges(EngineCase):
    def test_levels_grow_as_data_pushes_down(self):
        eng = self.open_engine(level_size_base=100, level_size_multiplier=2)
        for round_no in range(6):
            for i in range(4):
                eng.put(f"r{round_no}k{i}", "payload-" + "y" * 30)
            eng.flush()
            eng.auto_compact()
        deepest = max(
            (f.level for f in eng.version.all_files()),
            default=-1,
        )
        self.assertGreaterEqual(deepest, 2)
        for round_no in range(6):
            for i in range(4):
                self.assertEqual(eng.get(f"r{round_no}k{i}"), "payload-" + "y" * 30)

    def test_trivial_move_when_no_overlap_below(self):
        eng = self.open_engine()
        eng.inject_file(1, {"a": (1, "1"), "b": (2, "2")})
        eng.inject_file(1, {"x": (3, "3"), "y": (4, "4"), "z": (5, "w" * 150)})
        task = eng.pick_compaction()
        self.assertIsNotNone(task)
        self.assertEqual(task.kind, KIND_TRIVIAL_MOVE)
        handle = eng.run_compaction(task)
        self.assertEqual(handle.outputs[0].level, 2)
        self.assertEqual(handle.outputs[0].file_id, task.inputs[0].file_id)
        self.assertEqual(eng.get("x"), "3")


class TestDeterminism(EngineCase):
    def script(self, eng):
        for i in range(8):
            eng.put(f"key{i % 5}", f"val{i}")
            if i % 3 == 0:
                eng.delete(f"key{(i + 1) % 5}")
            if i % 2 == 0:
                eng.flush()
        eng.compact_all()

    def state_dump(self, eng):
        return [
            (m.to_dict(), eng._entries_of(m.file_id))
            for m in eng.version.all_files()
        ]

    def test_same_operations_same_result(self):
        with tempfile.TemporaryDirectory() as dir_a, tempfile.TemporaryDirectory() as dir_b:
            eng_a = LSMEngine.open(dir_a, small_options())
            eng_b = LSMEngine.open(dir_b, small_options())
            self.script(eng_a)
            self.script(eng_b)
            self.assertEqual(self.state_dump(eng_a), self.state_dump(eng_b))

    def test_picker_is_deterministic(self):
        eng = self.open_engine()
        for i in range(3):
            eng.put(f"k{i}", "v")
            eng.flush()
        first = eng.pick_compaction()
        second = eng.pick_compaction()
        self.assertEqual(first.inputs, second.inputs)
        self.assertEqual(first.grandparents, second.grandparents)
        self.assertEqual(first.output_level, second.output_level)


class TestWriteAmplification(EngineCase):
    def test_stats_track_compaction_io(self):
        eng = self.open_engine()
        for i in range(4):
            eng.put(f"k{i}", "v" * 10)
            eng.flush()
        eng.compact_all()
        self.assertGreater(eng.stats["compactions"], 0)
        self.assertGreater(eng.stats["compaction_bytes_read"], 0)
        self.assertGreater(eng.stats["compaction_bytes_written"], 0)
        self.assertLessEqual(
            eng.stats["compaction_bytes_written"],
            eng.stats["compaction_bytes_read"],
        )


if __name__ == "__main__":
    unittest.main()
