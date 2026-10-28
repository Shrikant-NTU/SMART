from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any


class VideoAnalyzer:
    """Extracts visual frames from archived MP4 files, falling back to PNGs.

    Set `video_frame_mode: "target_video_seconds"` to save a fraction of the
    matching example script's target_video_seconds as representative frames.
    Set `video_frame_mode: "all"` to save one PNG per actual video frame.
    Set `video_frame_mode: "sample"` to save at most `max_video_frames`.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.enabled = bool(config.get("enable_video_analysis", True))
        self.frame_mode = str(config.get("video_frame_mode", "sample")).lower()
        self.max_frames = int(config.get("max_video_frames", 6))
        self.target_seconds_fraction = float(config.get("target_video_seconds_frame_fraction", 0.5))

    def analyze_outputs(self, output_dir: Path, workspace: Path | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "frames": [], "summary": "Video analysis disabled."}

        images_dir = output_dir / "images"
        frames_root = output_dir / "frames"
        frames_root.mkdir(parents=True, exist_ok=True)
        videos = sorted(images_dir.glob(self.config.get("output_video_glob", "*.mp4")))

        extracted: list[str] = []
        video_metadata: list[dict[str, Any]] = []
        errors: list[str] = []
        for video in videos:
            try:
                target_frame_count = self._target_frame_count_for_video(video, workspace)
                frame_result = self.extract_frames(
                    video,
                    frames_root / video.stem,
                    target_frame_count=target_frame_count,
                )
                extracted.extend(frame_result["frames"])
                if frame_result.get("error"):
                    errors.append(f"{video.name}: {frame_result['error']}")
                video_metadata.append(
                    {
                        "video": str(video),
                        "frame_count": frame_result["frame_count"],
                        "fps": frame_result["fps"],
                        "extracted_frame_count": len(frame_result["frames"]),
                        "target_frame_count": target_frame_count,
                        "frame_mode": self.frame_mode,
                        "target_video_seconds_frame_fraction": self.target_seconds_fraction,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - persisted for diagnosis
                errors.append(f"{video.name}: {exc!r}")

        fallback_images = []
        if not extracted:
            fallback_images = self._fallback_image_frames(images_dir)
            extracted.extend(str(path) for path in fallback_images)

        result = {
            "enabled": True,
            "video_count": len(videos),
            "frames": extracted,
            "video_metadata": video_metadata,
            "fallback_image_count": len(fallback_images),
            "fallback_images": [str(path) for path in fallback_images],
            "errors": errors,
            "summary": self._summary(len(extracted), len(videos), len(fallback_images)),
        }
        (output_dir / "video_analysis.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        return result

    def extract_frames(
        self,
        video_path: Path,
        destination_dir: Path,
        target_frame_count: int | None = None,
    ) -> dict[str, Any]:
        try:
            import cv2  # type: ignore
        except ImportError:
            return {
                "frames": [],
                "frame_count": 0,
                "fps": None,
                "error": "OpenCV cv2 is not installed in the active Python environment.",
            }

        destination_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return {
                "frames": [],
                "frame_count": 0,
                "fps": None,
                "error": "OpenCV could not open the MP4 file.",
            }

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            capture.release()
            return {
                "frames": [],
                "frame_count": 0,
                "fps": fps,
                "error": "OpenCV opened the MP4 but reported zero frames.",
            }

        if self.frame_mode == "all":
            indices = list(range(frame_count))
        else:
            count = target_frame_count if target_frame_count is not None else self.max_frames
            count = min(max(1, int(count)), frame_count)
            indices = sorted({round(i * (frame_count - 1) / max(count - 1, 1)) for i in range(count)})

        saved: list[str] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            output = destination_dir / f"{video_path.stem}_frame_{index:05d}.png"
            cv2.imwrite(str(output), frame)
            saved.append(str(output))

        capture.release()
        return {"frames": saved, "frame_count": frame_count, "fps": fps}

    def _target_frame_count_for_video(self, video_path: Path, workspace: Path | None) -> int | None:
        if self.frame_mode != "target_video_seconds":
            return None
        if workspace is None:
            return self.max_frames

        script_path = workspace / f"{video_path.stem}.py"
        value = self._read_target_video_seconds(script_path)
        if value is None:
            return self.max_frames
        return max(1, int(round(value * self.target_seconds_fraction)))

    @staticmethod
    def _read_target_video_seconds(script_path: Path) -> float | None:
        if not script_path.exists():
            return None
        text = script_path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None

        if tree is not None:
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                if not any(isinstance(target, ast.Name) and target.id == "target_video_seconds" for target in node.targets):
                    continue
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                    return float(node.value.value)

        match = re.search(r"^\s*target_video_seconds\s*=\s*([0-9]+(?:\.[0-9]+)?)", text, re.MULTILINE)
        if not match:
            return None
        return float(match.group(1))

    def _fallback_image_frames(self, images_dir: Path) -> list[Path]:
        if not images_dir.is_dir():
            return []
        images = [
            path
            for path in sorted(images_dir.glob(self.config.get("output_image_glob", "*.png")))
            if path.is_file()
        ]
        if self.frame_mode == "all" or len(images) <= self.max_frames:
            return images
        count = min(max(1, self.max_frames), len(images))
        indices = sorted({round(i * (len(images) - 1) / max(count - 1, 1)) for i in range(count)})
        return [images[index] for index in indices]

    @staticmethod
    def _summary(frame_count: int, video_count: int, fallback_image_count: int) -> str:
        if fallback_image_count:
            return (
                f"Using {fallback_image_count} PNG image(s) as visual frame fallback; "
                f"extracted {frame_count - fallback_image_count} frame(s) from {video_count} video(s)."
            )
        return f"Extracted {frame_count} frame(s) from {video_count} video(s)."
