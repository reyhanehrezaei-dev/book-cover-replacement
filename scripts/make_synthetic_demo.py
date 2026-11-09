"""Create deterministic, original synthetic inputs and outputs for the GitHub demo.

No movie trailers, publisher covers, or third-party images are used.
"""
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cover_replacer import process_image, process_video  # noqa: E402

DEMO = ROOT / "examples" / "synthetic"
DEMO.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(2026)


def make_cover() -> np.ndarray:
    h, w = 460, 340
    y, x = np.indices((h, w))
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[:, :, 0] = np.clip(100 + 35 * np.sin(x / 17) + 20 * np.cos(y / 12), 0, 255)
    base[:, :, 1] = np.clip(65 + 30 * np.cos(x / 13) + 24 * np.sin(y / 18), 0, 255)
    base[:, :, 2] = np.clip(35 + 23 * np.sin((x + y) / 25), 0, 255)
    for i in range(145):
        cx = int(RNG.integers(18, w - 18))
        cy = int(RNG.integers(18, h - 18))
        rad = int(RNG.integers(2, 11))
        color = tuple(int(v) for v in RNG.integers(100, 245, 3))
        cv2.circle(base, (cx, cy), rad, color, 1, cv2.LINE_AA)
    cv2.rectangle(base, (9, 9), (w - 10, h - 10), (245, 229, 190), 4)
    cv2.rectangle(base, (18, 18), (w - 19, h - 19), (210, 185, 145), 1)
    cv2.putText(base, "FIELD GUIDE", (30, 64), cv2.FONT_HERSHEY_DUPLEX,
                0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(base, "VOL. 01", (107, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.67, (235, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(base, "VISION", (53, 398), cv2.FONT_HERSHEY_DUPLEX,
                1.1, (245, 247, 248), 2, cv2.LINE_AA)
    return base


def make_replacement(t: int = 0) -> np.ndarray:
    h, w = 460, 340
    y, x = np.indices((h, w))
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :, 0] = np.clip(40 + 50 * x / w, 0, 255)
    out[:, :, 1] = np.clip(36 + 75 * y / h, 0, 255)
    out[:, :, 2] = np.clip(125 + 70 * (1 - y / h), 0, 255)
    cx = 170 + int(50 * np.sin(t / 4))
    cy = 240 + int(40 * np.cos(t / 4))
    for r in (110, 86, 62):
        cv2.circle(out, (cx, cy), r, (240 - r // 2, 185 + r // 5, 225), 3, cv2.LINE_AA)
    cv2.putText(out, "NOVA", (75, 82), cv2.FONT_HERSHEY_DUPLEX,
                1.55, (248, 247, 246), 3, cv2.LINE_AA)
    cv2.putText(out, "A SYNTHETIC DEMO", (50, 420), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (240, 237, 220), 2, cv2.LINE_AA)
    return out


def make_scene(cover: np.ndarray, frame_no: int = 0, size=(840, 660)) -> np.ndarray:
    w, h = size
    y, x = np.indices((h, w))
    b = np.clip(60 + x * 45 / w + y * 25 / h, 0, 255).astype(np.uint8)
    scene = np.stack([b, np.minimum(b + 17, 255), np.minimum(b + 27, 255)], axis=-1)
    cv2.rectangle(scene, (0, h - 95), (w, h), (67, 84, 109), -1)
    cv2.putText(scene, "CAMERA FRAME / SYNTHETIC SCENE", (40, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 216, 222), 2, cv2.LINE_AA)
    rw, rh = cover.shape[1], cover.shape[0]
    dx = int(12 * np.sin(frame_no / 5))
    dy = int(8 * np.cos(frame_no / 6))
    source = np.float32([[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]])
    target = np.float32([[225 + dx, 72 + dy], [547 + dx, 97 + dy],
                         [568 + dx, 575 + dy], [205 + dx, 562 + dy]])
    H = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(cover, H, (w, h))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, target.astype(np.int32), 255)
    return np.where(mask[..., None] > 0, warped, scene).astype(np.uint8)


def write_clip(path: Path, frames: list[np.ndarray], fps: float = 12.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write MP4: {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def main() -> None:
    cover = make_cover()
    replacement = make_replacement()
    scene = make_scene(cover)
    after, match = process_image(scene, cover, replacement)
    if match is None:
        raise RuntimeError("Synthetic cover matching failed")
    files = {
        "cover_reference.png": cover,
        "replacement_art.png": replacement,
        "scene_before.png": scene,
        "scene_after.png": after,
    }
    for name, frame in files.items():
        if not cv2.imwrite(str(DEMO / name), frame):
            raise RuntimeError(f"Could not save {name}")
    print(f"Synthetic still image: matches={match.good_matches}, inliers={match.inliers}")

    scene_frames = [make_scene(cover, i) for i in range(12)]
    replacement_frames = [make_replacement(i) for i in range(12)]
    write_clip(DEMO / "input_clip.mp4", scene_frames)
    write_clip(DEMO / "overlay_clip.mp4", replacement_frames)
    count, hits = process_video(DEMO / "input_clip.mp4", cover,
                                DEMO / "overlay_clip.mp4", DEMO / "output_clip.mp4")
    if count < 10 or hits < 10:
        raise RuntimeError(f"Synthetic video test failed: frames={count}, detections={hits}")
    print(f"Synthetic video: frames={count}, detected_frames={hits}")


if __name__ == "__main__":
    main()
