from __future__ import annotations

import glob
import os
import re
from dataclasses import replace
from typing import Dict, List, Optional, Set, Tuple

from .compaction import (
    KIND_DELETE_EMPTY,
    KIND_MERGE,
    KIND_TRIVIAL_MOVE,
    CompactionPicker,
    CompactionTask,
    execute_merge,
)
from .errors import CompactionError
from .manifest import WriteAheadLog
from .options import Options
from .sstable import Entry, FileMeta, build_meta, entry_size, read_sst_file, write_sst_file
from .version import VersionEdit, VersionSet


class CompactionHandle:
    def __init__(self, task: CompactionTask):
        self.task = task
        self.outputs: List[FileMeta] = []
        self.output_paths: List[str] = []


class LSMEngine:
    def __init__(self, directory: str, options: Optional[Options] = None):
        self.options = options or Options()
        self.directory = directory
        self.sst_dir = os.path.join(directory, "sst")
        os.makedirs(self.sst_dir, exist_ok=True)

        self._vs = VersionSet.recover(directory, self.options)
        self._picker = CompactionPicker(self.options)
        self._wal = WriteAheadLog(directory)
        self._memtable: Dict[str, Entry] = {}
        self._memtable_size = 0
        self._entries_cache: Dict[int, Dict[str, Entry]] = {}
        self._in_progress: Dict[int, CompactionHandle] = {}
        self.stats = {
            "compactions": 0,
            "compaction_bytes_read": 0,
            "compaction_bytes_written": 0,
            "tombstones_dropped": 0,
        }

        self._remove_orphan_ssts()
        self._replay_wal()
        self._vs.snapshot()

    # ------------------------------------------------------------------ open

    @classmethod
    def open(cls, directory: str, options: Optional[Options] = None) -> "LSMEngine":
        return cls(directory, options)

    def _remove_orphan_ssts(self) -> None:
        referenced = {f.file_id for f in self._vs.current.all_files()}
        for path in glob.glob(os.path.join(self.sst_dir, "*.sst")):
            match = re.search(r"(\d+)\.sst$", os.path.basename(path))
            if match and int(match.group(1)) not in referenced:
                os.remove(path)

    def _replay_wal(self) -> None:
        for record in self._wal.read():
            seq = record["seq"]
            self._memtable[record["key"]] = (seq, record["value"])
            self._memtable_size += entry_size(record["key"], record["value"])
            if seq > self._vs.last_seq:
                self._vs.last_seq = seq

    # -------------------------------------------------------------- mutation

    def put(self, key: str, value: str) -> None:
        self._apply_op(key, value)

    def delete(self, key: str) -> None:
        self._apply_op(key, None)

    def _apply_op(self, key: str, value: Optional[str]) -> None:
        seq = self._vs.alloc_seq()
        self._wal.append({"key": key, "value": value, "seq": seq})
        old = self._memtable.get(key)
        if old is not None:
            self._memtable_size -= entry_size(key, old[1])
        self._memtable[key] = (seq, value)
        self._memtable_size += entry_size(key, value)
        if self._memtable_size >= self.options.memtable_size_limit:
            self.flush()

    def flush(self) -> Optional[FileMeta]:
        if not self._memtable:
            return None
        entries = dict(self._memtable)
        file_id = self._vs.alloc_file_id()
        meta = build_meta(file_id, 0, entries)
        self._write_sst(meta, entries)
        self._vs.log_and_apply(VersionEdit(add=[meta]))
        self._entries_cache[file_id] = entries
        self._memtable.clear()
        self._memtable_size = 0
        self._wal.reset()
        return meta

    # ---------------------------------------------------------------- reads

    def get(self, key: str) -> Optional[str]:
        hit = self._memtable.get(key)
        if hit is not None:
            return hit[1]
        for meta in self._vs.current.visible_order():
            if not meta.covers(key):
                continue
            entries = self._entries_of(meta.file_id)
            if key in entries:
                return entries[key][1]
        return None

    def scan(self, start: Optional[str] = None, end: Optional[str] = None) -> List[Tuple[str, str]]:
        best: Dict[str, Entry] = {}
        sources: List[Dict[str, Entry]] = [self._memtable]
        for meta in self._vs.current.all_files():
            if meta.empty:
                continue
            sources.append(self._entries_of(meta.file_id))
        for entries in sources:
            for key, entry in entries.items():
                if start is not None and key < start:
                    continue
                if end is not None and key > end:
                    continue
                current = best.get(key)
                if current is None or entry[0] > current[0]:
                    best[key] = entry
        return sorted(
            (key, value) for key, (seq, value) in best.items() if value is not None
        )

    # ------------------------------------------------------------ compaction

    def pick_compaction(self) -> Optional[CompactionTask]:
        return self._picker.pick(self._vs.current, self._busy_file_ids())

    def begin_compaction(self, task: CompactionTask) -> CompactionHandle:
        busy = self._busy_file_ids()
        conflict = task.all_input_ids() & busy
        if conflict:
            raise CompactionError(f"files already compacting: {sorted(conflict)}")
        missing = [fid for fid in task.all_input_ids() if self._vs.current.find(fid) is None]
        if missing:
            raise CompactionError(f"input files no longer live: {sorted(missing)}")
        handle = CompactionHandle(task)
        self._in_progress[task.task_id] = handle
        return handle

    def commit_compaction(self, handle: CompactionHandle) -> None:
        task = handle.task
        edit = VersionEdit(
            add=list(handle.outputs),
            delete=[(f.level, f.file_id) for f in list(task.inputs) + list(task.grandparents)],
        )
        self._vs.log_and_apply(edit)
        kept = {m.file_id for m in handle.outputs}
        for meta in list(task.inputs) + list(task.grandparents):
            if meta.file_id not in kept:
                self._remove_sst(meta.file_id)
        self._in_progress.pop(task.task_id, None)
        self.stats["compactions"] += 1

    def abort_compaction(self, handle: CompactionHandle) -> None:
        for path in handle.output_paths:
            if os.path.exists(path):
                os.remove(path)
        for meta in handle.outputs:
            self._entries_cache.pop(meta.file_id, None)
        self._in_progress.pop(handle.task.task_id, None)

    def run_compaction(
        self,
        task: Optional[CompactionTask] = None,
        fail_after_writes: Optional[int] = None,
        commit: bool = True,
    ) -> Optional[CompactionHandle]:
        if task is None:
            task = self.pick_compaction()
        if task is None:
            return None
        handle = self.begin_compaction(task)
        try:
            self._execute(handle, fail_after_writes)
        except Exception:
            self.abort_compaction(handle)
            raise
        if commit:
            self.commit_compaction(handle)
        return handle

    def auto_compact(self) -> None:
        while True:
            self.flush()
            if self.run_compaction() is None:
                return

    def compact_all(self) -> None:
        """Full major compaction: scored tasks first, then forced level pushes."""
        while True:
            self.flush()
            task = self.pick_compaction()
            if task is None:
                task = self._picker.pick_forced(self._vs.current, self._busy_file_ids())
            if task is None:
                return
            self.run_compaction(task)

    def _execute(self, handle: CompactionHandle, fail_after_writes: Optional[int]) -> None:
        task = handle.task
        if task.kind == KIND_DELETE_EMPTY:
            return
        if task.kind == KIND_TRIVIAL_MOVE:
            handle.outputs = [
                replace(f, level=task.output_level) for f in task.inputs
            ]
            return

        deeper = self._vs.current.files_deeper_than(task.output_level)
        groups = execute_merge(task, self._entries_of, deeper, self.options)
        read_bytes = sum(f.size for f in list(task.inputs) + list(task.grandparents))
        self.stats["compaction_bytes_read"] += read_bytes
        written = 0
        for group in groups:
            file_id = self._vs.alloc_file_id()
            meta = build_meta(file_id, task.output_level, group)
            path = self._write_sst(meta, group)
            handle.outputs.append(meta)
            handle.output_paths.append(path)
            self._entries_cache[file_id] = group
            self.stats["compaction_bytes_written"] += meta.size
            written += 1
            if fail_after_writes is not None and written >= fail_after_writes:
                raise CompactionError("injected compaction failure")
        dropped = sum(len(f.tombstones) for f in list(task.inputs) + list(task.grandparents))
        dropped -= sum(len(m.tombstones) for m in handle.outputs)
        self.stats["tombstones_dropped"] += max(dropped, 0)

    # ---------------------------------------------------------------- files

    def _sst_path(self, file_id: int) -> str:
        return os.path.join(self.sst_dir, f"{file_id:06d}.sst")

    def _write_sst(self, meta: FileMeta, entries: Dict[str, Entry]) -> str:
        path = self._sst_path(meta.file_id)
        write_sst_file(path, meta, entries)
        return path

    def _remove_sst(self, file_id: int) -> None:
        self._entries_cache.pop(file_id, None)
        path = self._sst_path(file_id)
        if os.path.exists(path):
            os.remove(path)

    def _entries_of(self, file_id: int) -> Dict[str, Entry]:
        if file_id not in self._entries_cache:
            _, entries = read_sst_file(self._sst_path(file_id))
            self._entries_cache[file_id] = entries
        return self._entries_cache[file_id]

    def _busy_file_ids(self) -> Set[int]:
        busy: Set[int] = set()
        for handle in self._in_progress.values():
            busy |= handle.task.all_input_ids()
        return busy

    # ----------------------------------------------------------------- misc

    @property
    def version(self):
        return self._vs.current

    def inject_file(self, level: int, entries: Dict[str, Entry]) -> FileMeta:
        """Test helper: install an SST directly at a level via the manifest."""
        file_id = self._vs.alloc_file_id()
        meta = build_meta(file_id, level, entries)
        if meta.generation > self._vs.last_seq:
            self._vs.last_seq = meta.generation
        if entries:
            self._write_sst(meta, entries)
            self._entries_cache[file_id] = entries
        self._vs.log_and_apply(VersionEdit(add=[meta]))
        return meta

    def close(self) -> None:
        self._vs.snapshot()
