from dataclasses import dataclass


@dataclass(frozen=True)
class Options:
    max_levels: int = 7
    l0_compaction_trigger: int = 4
    max_file_size: int = 2048
    level_size_base: int = 4096
    level_size_multiplier: int = 10
    memtable_size_limit: int = 4096

    def level_capacity(self, level: int) -> int:
        if level <= 0:
            return self.l0_compaction_trigger
        return self.level_size_base * (self.level_size_multiplier ** (level - 1))
