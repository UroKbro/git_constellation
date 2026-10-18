"""Build Commit objects without git."""
from constellation.types import Commit, FileChange
from tests.repo_builder import DAY, T0


def c(i: int, changes: list[FileChange], author: str = "Ana",
      parents: tuple[str, ...] | None = None, day: float | None = None) -> Commit:
    if parents is None:
        parents = (f"s{i-1}",) if i > 0 else ()
    return Commit(sha=f"s{i}", parents=parents, author=author,
                  email=f"{author.lower()}@example.com",
                  ts=T0 + int((i if day is None else day) * DAY),
                  subject=f"c{i}", changes=tuple(changes))


def A(path: str, added: int, binary: bool = False) -> FileChange:
    return FileChange("A", path, None, 0 if binary else added, 0, binary)


def M(path: str, added: int, deleted: int) -> FileChange:
    return FileChange("M", path, None, added, deleted, False)


def D(path: str, deleted: int, binary: bool = False) -> FileChange:
    return FileChange("D", path, None, 0, 0 if binary else deleted, binary)


def R(old: str, new: str, added: int, deleted: int) -> FileChange:
    return FileChange("R", new, old, added, deleted, False)
