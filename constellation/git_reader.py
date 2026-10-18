"""Read-only access to a git repository via plumbing commands."""
import subprocess
from collections.abc import Iterator
from pathlib import Path

from constellation.types import Commit, FileChange, Tag

_LOG_FORMAT = "%x1e%H%x1f%P%x1f%aN%x1f%aE%x1f%at%x1f%s"


class NotARepoError(Exception):
    pass


class UnknownRefError(Exception):
    pass


def _git(root: Path, *args: str) -> list[str]:
    return ["git", "-C", str(root), "-c", "core.quotepath=off", *args]


def _dec(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")


def repo_root(path: Path) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise NotARepoError(f"not a directory: {path}")
    r = subprocess.run(_git(path, "rev-parse", "--show-toplevel"), capture_output=True)
    if r.returncode != 0:
        raise NotARepoError(f"not a git repository: {path}")
    return Path(_dec(r.stdout).strip()).resolve()


def head_sha(root: Path, ref: str = "HEAD") -> str | None:
    r = subprocess.run(_git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"),
                       capture_output=True)
    if r.returncode == 0:
        return _dec(r.stdout).strip()
    any_commit = subprocess.run(_git(root, "rev-list", "-n", "1", "--all"),
                                capture_output=True)
    if any_commit.returncode == 0 and not any_commit.stdout.strip():
        return None
    raise UnknownRefError(ref)


def parse_log(data: bytes) -> Iterator[Commit]:
    for record in data.split(b"\x1e"):
        if not record.strip(b"\n"):
            continue
        header, _, body = record.partition(b"\0")
        fields = _dec(header).split("\x1f", 5)
        sha, parents, author, email, ts, subject = fields
        tokens = body.split(b"\0")
        if tokens:
            tokens[0] = tokens[0].lstrip(b"\n")
        i = 0
        raw: list[tuple[str, str, str | None]] = []
        while i < len(tokens) and tokens[i].startswith(b":"):
            status = _dec(tokens[i]).rsplit(" ", 1)[-1][0]
            if status in ("R", "C"):
                raw.append((status, _dec(tokens[i + 2]), _dec(tokens[i + 1])))
                i += 3
            else:
                raw.append((status, _dec(tokens[i + 1]), None))
                i += 2
        changes = []
        for status, path, old in raw:
            parts = _dec(tokens[i]).split("\t")
            i += 3 if status in ("R", "C") else 1
            binary = parts[0] == "-"
            added = 0 if binary else int(parts[0])
            deleted = 0 if binary else int(parts[1])
            changes.append(FileChange(status, path, old, added, deleted, binary))
        yield Commit(sha, tuple(parents.split()), author, email, int(ts), subject,
                     tuple(changes))


def iter_commits(root: Path, ref: str = "HEAD", exclude: str | None = None) -> Iterator[Commit]:
    spec = f"{exclude}..{ref}" if exclude else ref
    cmd = _git(root, "log", "--reverse", "--date-order", "--find-renames", "--raw",
               "--numstat", "-z", f"--format={_LOG_FORMAT}", spec)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if proc.returncode != 0:
        raise UnknownRefError(_dec(err).strip() or spec)
    yield from parse_log(out)


def read_tags(root: Path) -> list[Tag]:
    r = subprocess.run(
        _git(root, "for-each-ref", "refs/tags",
             "--format=%(refname:short)%09%(objectname)%09%(*objectname)"),
        capture_output=True, check=True)
    tags = []
    for line in _dec(r.stdout).splitlines():
        name, obj, peeled = (line.split("\t") + ["", ""])[:3]
        tags.append(Tag(name, peeled or obj))
    return tags


def head_loc(root: Path, ref: str) -> dict[str, int]:
    ls = subprocess.run(_git(root, "ls-tree", "-r", "-z", ref), capture_output=True,
                        check=True)
    entries = []  # (path, sha)
    for rec in ls.stdout.split(b"\0"):
        if not rec:
            continue
        meta, _, path = rec.partition(b"\t")
        mode, kind, sha = meta.split(b" ")
        if kind == b"blob":
            entries.append((_dec(path), sha))
    if not entries:
        return {}
    stdin = b"".join(sha + b"\n" for _, sha in entries)
    proc = subprocess.run(_git(root, "cat-file", "--batch"), input=stdin,
                          capture_output=True, check=True)
    out, pos, result = proc.stdout, 0, {}
    for path, _ in entries:
        nl = out.index(b"\n", pos)
        size = int(out[pos:nl].split(b" ")[2])
        data = out[nl + 1:nl + 1 + size]
        pos = nl + 1 + size + 1
        if b"\0" in data[:8000]:
            result[path] = 0
        else:
            result[path] = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
    return result
