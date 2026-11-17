from __future__ import annotations

import json
import os
from typing import List

MANIFEST_NAME = "MANIFEST"
WAL_NAME = "wal.log"


def _append_line(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, sort_keys=True) + "\n")
        fp.flush()
        os.fsync(fp.fileno())


def _read_lines(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class Manifest:
    """Append-only log of version edits; rewritten atomically on snapshot."""

    def __init__(self, directory: str):
        self.path = os.path.join(directory, MANIFEST_NAME)

    def append(self, record: dict) -> None:
        _append_line(self.path, record)

    def read(self) -> List[dict]:
        return _read_lines(self.path)

    def rewrite(self, records: List[dict]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            for record in records:
                fp.write(json.dumps(record, sort_keys=True) + "\n")
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, self.path)


class WriteAheadLog:
    def __init__(self, directory: str):
        self.path = os.path.join(directory, WAL_NAME)

    def append(self, record: dict) -> None:
        _append_line(self.path, record)

    def read(self) -> List[dict]:
        return _read_lines(self.path)

    def reset(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)
