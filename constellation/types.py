"""Shared immutable data types."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FileChange:
    status: str  # "A" | "M" | "D" | "R" | "T"
    path: str
    old_path: str | None
    added: int
    deleted: int
    binary: bool


@dataclass(frozen=True, slots=True)
class Commit:
    sha: str
    parents: tuple[str, ...]
    author: str
    email: str
    ts: int
    subject: str
    changes: tuple[FileChange, ...]


@dataclass(frozen=True, slots=True)
class Tag:
    name: str
    sha: str  # peeled commit sha
