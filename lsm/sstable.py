from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

Entry = Tuple[int, Optional[str]]

ENTRY_OVERHEAD = 16


def entry_size(key: str, value: Optional[str]) -> int:
    return ENTRY_OVERHEAD + len(key) + (len(value) if value is not None else 0)


@dataclass(frozen=True)
class FileMeta:
    file_id: int
    level: int
    smallest: Optional[str]
    largest: Optional[str]
    size: int
    generation: int
    tombstones: frozenset

    @property
    def empty(self) -> bool:
        return self.smallest is None

    def overlaps(self, smallest: str, largest: str) -> bool:
        if self.empty:
            return False
        return self.smallest <= largest and smallest <= self.largest

    def covers(self, key: str) -> bool:
        return not self.empty and self.smallest <= key <= self.largest

    def to_dict(self) -> dict:
        return {
            "file_id": self.file_id,
            "level": self.level,
            "smallest": self.smallest,
            "largest": self.largest,
            "size": self.size,
            "generation": self.generation,
            "tombstones": sorted(self.tombstones),
        }

    @staticmethod
    def from_dict(data: dict) -> "FileMeta":
        return FileMeta(
            file_id=data["file_id"],
            level=data["level"],
            smallest=data["smallest"],
            largest=data["largest"],
            size=data["size"],
            generation=data["generation"],
            tombstones=frozenset(data["tombstones"]),
        )


def build_meta(file_id: int, level: int, entries: Dict[str, Entry]) -> FileMeta:
    if entries:
        keys = sorted(entries)
        smallest, largest = keys[0], keys[-1]
        generation = max(seq for seq, _ in entries.values())
    else:
        smallest = largest = None
        generation = 0
    size = sum(entry_size(k, v) for k, (_, v) in entries.items())
    tombstones = frozenset(k for k, (_, v) in entries.items() if v is None)
    return FileMeta(
        file_id=file_id,
        level=level,
        smallest=smallest,
        largest=largest,
        size=size,
        generation=generation,
        tombstones=tombstones,
    )


def write_sst_file(path: str, meta: FileMeta, entries: Dict[str, Entry]) -> None:
    payload = {
        "meta": meta.to_dict(),
        "entries": {k: [seq, val] for k, (seq, val) in sorted(entries.items())},
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, sort_keys=True)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, path)


def read_sst_file(path: str) -> Tuple[FileMeta, Dict[str, Entry]]:
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    meta = FileMeta.from_dict(payload["meta"])
    entries = {k: (seq, val) for k, (seq, val) in payload["entries"].items()}
    return meta, entries
