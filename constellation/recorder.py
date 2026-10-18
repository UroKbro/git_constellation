"""Screenshots and video: PNG files, and MP4 / GIF streamed to ffmpeg (PNG frames without it).

Frames are (h, w, 3) uint8 arrays, top row first. A writer thread feeds ffmpeg so the
render loop never waits on the encoder."""
from __future__ import annotations

import queue
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

MP4_WIDTH = 1920
GIF_WIDTH = 960
GIF_FPS = 15
QUEUE_FRAMES = 8


def output_dir() -> Path:
    """~/Pictures/git_constellation, or the current directory if that can't be created."""
    d = Path.home() / "Pictures" / "git_constellation"
    try:
        d.mkdir(parents=True, exist_ok=True)
        return d
    except OSError:
        return Path.cwd()


def output_path(repo: str, ext: str, when: datetime | None = None,
                folder: Path | None = None) -> Path:
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in repo) or "repo"
    return (folder or output_dir()) / f"{safe}-{stamp}.{ext}"


def ffmpeg_cmd(ffmpeg: str, path: Path, size: tuple[int, int], fps: float) -> list[str]:
    """Command reading raw RGB frames of `size` from stdin; format from the suffix."""
    w, h = size
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-"]
    if path.suffix.lower() == ".gif":
        gw = min(GIF_WIDTH, w)
        cmd += ["-vf", f"fps={min(GIF_FPS, fps):g},scale={gw}:-1:flags=lanczos,split[a][b];"
                       "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                "-loop", "0"]
    else:
        mw = min(MP4_WIDTH, w) // 2 * 2
        cmd += ["-vf", f"scale={mw}:-2:flags=lanczos", "-c:v", "libx264", "-preset", "medium",
                "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    return cmd + [str(path)]


def save_png(pixels: np.ndarray, path: Path) -> Path:
    import pygame
    h, w = pixels.shape[:2]
    pygame.image.save(pygame.image.frombytes(np.ascontiguousarray(pixels).tobytes(),
                                             (w, h), "RGB"), str(path))
    return path


class VideoWriter:
    """Streams frames to ffmpeg, or to a PNG frame folder when ffmpeg isn't on PATH.

    `path` is the requested file; `result` is what was actually written."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float,
                 ffmpeg: str | None = None) -> None:
        self.path, self.size, self.fps = Path(path), tuple(size), fps
        self.frames = 0
        self.error: str | None = None
        ffmpeg = ffmpeg if ffmpeg is not None else shutil.which("ffmpeg")
        self._proc = None
        if ffmpeg:
            self.result = self.path
            self._proc = subprocess.Popen(ffmpeg_cmd(ffmpeg, self.path, self.size, fps),
                                          stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            self.result = self.path.with_suffix("")
            self.result.mkdir(parents=True, exist_ok=True)
        self._q: queue.Queue = queue.Queue(maxsize=QUEUE_FRAMES)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def write(self, pixels: np.ndarray) -> None:
        if pixels.shape[1::-1] != self.size:
            raise ValueError(f"frame {pixels.shape[1::-1]} != video {self.size}")
        self._q.put(np.ascontiguousarray(pixels[:, :, :3]))

    def _loop(self) -> None:
        while (frame := self._q.get()) is not None:
            if self.error is not None:
                continue
            try:
                if self._proc is not None:
                    self._proc.stdin.write(frame.tobytes())
                else:
                    save_png(frame, self.result / f"frame_{self.frames:05d}.png")
                self.frames += 1
            except (OSError, ValueError) as e:
                self.error = str(e)

    def close(self) -> Path | None:
        """Finish the file. Returns its path, or None if nothing usable was written."""
        self._q.put(None)
        self._thread.join()
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            err = self._proc.stderr.read().decode(errors="replace").strip()
            if self._proc.wait() != 0:
                self.error = self.error or err or f"ffmpeg exited {self._proc.returncode}"
        if self.error is not None or self.frames == 0:
            return None
        return self.result

    def abort(self) -> None:
        self._q.put(None)
        self._thread.join()
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
            self.path.unlink(missing_ok=True)
