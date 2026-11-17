from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .manifest import Manifest
from .options import Options
from .sstable import FileMeta


@dataclass
class VersionEdit:
    add: List[FileMeta] = field(default_factory=list)
    delete: List[Tuple[int, int]] = field(default_factory=list)  # (level, file_id)


class Version:
    def __init__(self, levels: List[Tuple[FileMeta, ...]]):
        ordered: List[Tuple[FileMeta, ...]] = []
        for idx, files in enumerate(levels):
            if idx == 0:
                ordered.append(tuple(sorted(files, key=lambda f: f.file_id)))
            else:
                key = lambda f: (f.smallest is None, f.smallest or "", f.file_id)
                ordered.append(tuple(sorted(files, key=key)))
        self.levels: Tuple[Tuple[FileMeta, ...], ...] = tuple(ordered)

    def files(self, level: int) -> Tuple[FileMeta, ...]:
        return self.levels[level]

    def all_files(self) -> List[FileMeta]:
        return [f for level in self.levels for f in level]

    def num_non_empty_levels(self) -> int:
        return sum(1 for level in self.levels if level)

    def find(self, file_id: int) -> Optional[FileMeta]:
        for f in self.all_files():
            if f.file_id == file_id:
                return f
        return None

    def files_deeper_than(self, level: int) -> List[FileMeta]:
        return [f for lvl in self.levels[level + 1:] for f in lvl]

    def visible_order(self) -> List[FileMeta]:
        """Read visibility order: L0 newest-first, then L1..Ln shallow to deep."""
        ordered = sorted(self.levels[0], key=lambda f: f.file_id, reverse=True)
        for level in self.levels[1:]:
            ordered.extend(level)
        return ordered


def apply_edit(version: Version, edit: VersionEdit, max_levels: int) -> Version:
    deleted: Dict[int, set] = {}
    for level, file_id in edit.delete:
        deleted.setdefault(level, set()).add(file_id)
    levels: List[List[FileMeta]] = []
    for idx, files in enumerate(version.levels):
        levels.append([f for f in files if f.file_id not in deleted.get(idx, set())])
    for meta in edit.add:
        levels[meta.level].append(meta)
    return Version([tuple(level) for level in levels])


class VersionSet:
    def __init__(self, directory: str, options: Options):
        self.options = options
        self.manifest = Manifest(directory)
        empty: List[Tuple[FileMeta, ...]] = [tuple() for _ in range(options.max_levels)]
        self.current = Version(empty)
        self.next_file_id = 1
        self.last_seq = 0

    def alloc_file_id(self) -> int:
        file_id = self.next_file_id
        self.next_file_id += 1
        return file_id

    def alloc_seq(self) -> int:
        self.last_seq += 1
        return self.last_seq

    def log_and_apply(self, edit: VersionEdit) -> None:
        self.manifest.append({
            "type": "edit",
            "add": [m.to_dict() for m in edit.add],
            "delete": [[lvl, fid] for lvl, fid in edit.delete],
            "next_file_id": self.next_file_id,
            "last_seq": self.last_seq,
        })
        self.current = apply_edit(self.current, edit, self.options.max_levels)

    def snapshot(self) -> None:
        self.manifest.rewrite([{
            "type": "snapshot",
            "files": [m.to_dict() for m in self.current.all_files()],
            "next_file_id": self.next_file_id,
            "last_seq": self.last_seq,
        }])

    @classmethod
    def recover(cls, directory: str, options: Options) -> "VersionSet":
        vs = cls(directory, options)
        files: Dict[int, FileMeta] = {}
        for record in vs.manifest.read():
            if record["type"] == "snapshot":
                files = {m["file_id"]: FileMeta.from_dict(m) for m in record["files"]}
                vs.next_file_id = record["next_file_id"]
                vs.last_seq = record["last_seq"]
            elif record["type"] == "edit":
                for m in record["add"]:
                    meta = FileMeta.from_dict(m)
                    files[meta.file_id] = meta
                for _level, file_id in record["delete"]:
                    files.pop(file_id, None)
                vs.next_file_id = record["next_file_id"]
                vs.last_seq = record["last_seq"]
        levels: List[List[FileMeta]] = [[] for _ in range(options.max_levels)]
        for meta in files.values():
            levels[meta.level].append(meta)
        vs.current = Version([tuple(level) for level in levels])
        return vs
