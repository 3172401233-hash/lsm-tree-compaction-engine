from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

from .options import Options
from .sstable import Entry, FileMeta, entry_size
from .version import Version

KIND_MERGE = "merge"
KIND_TRIVIAL_MOVE = "trivial_move"
KIND_DELETE_EMPTY = "delete_empty"


@dataclass(frozen=True)
class CompactionTask:
    task_id: int
    kind: str
    input_level: Optional[int]
    output_level: Optional[int]
    inputs: Tuple[FileMeta, ...]
    grandparents: Tuple[FileMeta, ...]

    def all_input_ids(self) -> Set[int]:
        return {f.file_id for f in self.inputs} | {f.file_id for f in self.grandparents}


def _range_of(files: Tuple[FileMeta, ...]) -> Tuple[str, str]:
    smallest = min(f.smallest for f in files if f.smallest is not None)
    largest = max(f.largest for f in files if f.largest is not None)
    return smallest, largest


class CompactionPicker:
    """Deterministic size-scored leveled picker.

    Scores: L0 by file count, deeper levels by total size vs. capacity.
    Highest score >= 1.0 wins; ties resolve to the shallower level.
    """

    def __init__(self, options: Options):
        self.options = options
        self._next_task_id = 1

    def pick(self, version: Version, in_progress: Set[int]) -> Optional[CompactionTask]:
        task = self._pick_empty_cleanup(version, in_progress)
        if task is not None:
            return task

        best_level = -1
        best_score = 1.0
        for level, files in enumerate(version.levels):
            available = [f for f in files if f.file_id not in in_progress]
            if not available:
                continue
            if level == 0:
                score = len(available) / self.options.l0_compaction_trigger
            else:
                score = sum(f.size for f in available) / self.options.level_capacity(level)
            if score >= 1.0 and (best_level < 0 or score > best_score):
                best_score = score
                best_level = level
        if best_level < 0:
            return None
        return self._build_task(version, best_level, in_progress)

    def pick_forced(self, version: Version, in_progress: Set[int]) -> Optional[CompactionTask]:
        """Major compaction: push the shallowest non-empty level one level down."""
        for level, files in enumerate(version.levels):
            available = tuple(f for f in files if f.file_id not in in_progress)
            if not available:
                continue
            if level >= self.options.max_levels - 1:
                return None
            output_level = level + 1
            smallest, largest = _range_of(available)
            grandparents = tuple(
                f for f in version.files(output_level)
                if f.overlaps(smallest, largest)
            )
            if any(f.file_id in in_progress for f in grandparents):
                return None
            if self._is_trivial(available, grandparents, output_level):
                kind = KIND_TRIVIAL_MOVE
            else:
                kind = KIND_MERGE
            return self._new_task(
                kind=kind,
                input_level=level,
                output_level=output_level,
                inputs=available,
                grandparents=grandparents,
            )
        return None

    def _is_trivial(self, inputs, grandparents, output_level) -> bool:
        if grandparents or len(inputs) != 1:
            return False
        if output_level == self.options.max_levels - 1 and inputs[0].tombstones:
            return False
        return True

    def _new_task(self, **kwargs) -> CompactionTask:
        task = CompactionTask(task_id=self._next_task_id, **kwargs)
        self._next_task_id += 1
        return task

    def _pick_empty_cleanup(self, version: Version, in_progress: Set[int]) -> Optional[CompactionTask]:
        empties = tuple(
            f for f in version.all_files()
            if f.empty and f.file_id not in in_progress
        )
        if not empties:
            return None
        return self._new_task(
            kind=KIND_DELETE_EMPTY,
            input_level=None,
            output_level=None,
            inputs=empties,
            grandparents=(),
        )

    def _build_task(self, version: Version, level: int, in_progress: Set[int]) -> Optional[CompactionTask]:
        output_level = min(level + 1, self.options.max_levels - 1)
        if level == 0:
            inputs = self._expand_l0(version, in_progress)
        else:
            inputs = (self._seed_file(version, level, in_progress),)
        smallest, largest = _range_of(inputs)

        grandparents = tuple(
            f for f in version.files(output_level)
            if f.file_id not in {i.file_id for i in inputs}
            and f.overlaps(smallest, largest)
        )
        if any(f.file_id in in_progress for f in grandparents):
            return None

        if output_level != level and self._is_trivial(inputs, grandparents, output_level):
            return self._new_task(
                kind=KIND_TRIVIAL_MOVE,
                input_level=level,
                output_level=output_level,
                inputs=inputs,
                grandparents=(),
            )
        return self._new_task(
            kind=KIND_MERGE,
            input_level=level,
            output_level=output_level,
            inputs=inputs,
            grandparents=grandparents,
        )

    def _seed_file(self, version: Version, level: int, in_progress: Set[int]) -> FileMeta:
        candidates = [f for f in version.files(level) if f.file_id not in in_progress]
        return max(candidates, key=lambda f: (f.size, -f.file_id))

    def _expand_l0(self, version: Version, in_progress: Set[int]) -> Tuple[FileMeta, ...]:
        available = [f for f in version.files(0) if f.file_id not in in_progress]
        seed = min(available, key=lambda f: f.file_id)
        selected = {seed.file_id: seed}
        smallest, largest = seed.smallest, seed.largest
        grew = True
        while grew:
            grew = False
            for f in available:
                if f.file_id in selected:
                    continue
                if f.overlaps(smallest, largest):
                    selected[f.file_id] = f
                    smallest = min(smallest, f.smallest)
                    largest = max(largest, f.largest)
                    grew = True
        return tuple(sorted(selected.values(), key=lambda f: f.file_id))


def execute_merge(
    task: CompactionTask,
    entries_of: Callable[[int], Dict[str, Entry]],
    deeper_files: List[FileMeta],
    options: Options,
) -> List[Dict[str, Entry]]:
    """Merge task inputs into output entry groups, newest sequence wins.

    A tombstone is physically dropped only when no file deeper than the
    output level can hold an older value for that key.
    """
    combined: Dict[str, Entry] = {}
    for meta in list(task.inputs) + list(task.grandparents):
        for key, entry in entries_of(meta.file_id).items():
            current = combined.get(key)
            if current is None or entry[0] > current[0]:
                combined[key] = entry

    def has_deeper_value(key: str) -> bool:
        return any(f.covers(key) for f in deeper_files)

    outputs: List[Dict[str, Entry]] = []
    current: Dict[str, Entry] = {}
    current_size = 0
    for key in sorted(combined):
        seq, value = combined[key]
        if value is None and not has_deeper_value(key):
            continue
        size = entry_size(key, value)
        if current and current_size + size > options.max_file_size:
            outputs.append(current)
            current = {}
            current_size = 0
        current[key] = (seq, value)
        current_size += size
    if current:
        outputs.append(current)
    return outputs
