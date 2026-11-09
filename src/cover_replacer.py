"""Feature-based planar book-cover replacement, adapted from the author's course notebooks.

Uses SIFT, FLANN K-D tree nearest-neighbor search, Lowe's ratio test, and
RANSAC homography. This is a classical computer-vision pipeline, not an ML model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    homography: np.ndarray
    corners: np.ndarray
    good_matches: int
    inliers: int
    area_fraction: float


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def locate_cover(
    frame: np.ndarray,
    reference_cover: np.ndarray,
    *,
    min_matches: int = 10,
    lowe_ratio: float = 0.7,
    min_area_fraction: float = 0.01,
    sift: cv2.SIFT | None = None,
) -> Detection | None:
    """Locate a known reference cover in a scene; None if evidence is insufficient."""
    if frame is None or reference_cover is None:
        return None
    sift = sift or cv2.SIFT_create()
    ref_gray = cv2.cvtColor(reference_cover, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ref_kp, ref_desc = sift.detectAndCompute(ref_gray, None)
    scene_kp, scene_desc = sift.detectAndCompute(scene_gray, None)
    if ref_desc is None or scene_desc is None or len(ref_desc) < 2 or len(scene_desc) < 2:
        return None

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    pairs = flann.knnMatch(ref_desc, scene_desc, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair]
            if m.distance < lowe_ratio * n.distance]
    if len(good) < max(4, min_matches):
        return None

    src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([scene_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or inlier_mask is None or not np.isfinite(H).all():
        return None
    inliers = int(inlier_mask.sum())
    if inliers < 4:
        return None

    height, width = reference_cover.shape[:2]
    rect = np.float32([[0, 0], [width - 1, 0],
                       [width - 1, height - 1], [0, height - 1]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(rect, H)
    if not np.isfinite(quad).all():
        return None
    quad_int = np.round(quad).astype(np.int32)
    if not cv2.isContourConvex(quad_int):
        return None
    area_fraction = abs(cv2.contourArea(quad)) / float(frame.shape[0] * frame.shape[1])
    if not (min_area_fraction <= area_fraction <= 1.0):
        return None
    return Detection(H, quad, len(good), inliers, area_fraction)


def replace_cover(
    frame: np.ndarray,
    reference_cover: np.ndarray,
    replacement: np.ndarray,
    detection: Detection,
    *,
    preserve_occlusion: bool = False,
    occlusion_threshold: int = 45,
) -> np.ndarray:
    """Warp an image/video frame onto the found cover's quadrilateral.

    Occlusion restoration is a heuristic based on reference-to-scene pixel
    differences; it is optional because lighting changes can produce artifacts.
    """
    height, width = frame.shape[:2]
    ref_height, ref_width = reference_cover.shape[:2]
    resized = cv2.resize(replacement, (ref_width, ref_height))
    warped = cv2.warpPerspective(resized, detection.homography, (width, height))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(detection.corners).astype(np.int32), 255)

    result = np.where(mask[..., None] > 0, warped, frame).astype(np.uint8)
    if not preserve_occlusion:
        return result

    warped_ref = cv2.warpPerspective(reference_cover, detection.homography, (width, height))
    ref_gray = cv2.cvtColor(warped_ref, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(ref_gray, scene_gray)
    _, occ = cv2.threshold(diff, occlusion_threshold, 255, cv2.THRESH_BINARY)
    occ = cv2.bitwise_and(occ, mask)
    kernel = np.ones((3, 3), np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, kernel)
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(occ)
    cover_area = max(1, int(np.count_nonzero(mask)))
    cleaned = np.zeros_like(occ)
    for label in range(1, num):
        area = stats[label, cv2.CC_STAT_AREA]
        if cover_area * 0.01 <= area <= cover_area * 0.5:
            cleaned[labels == label] = 255
    ratio = np.count_nonzero(cleaned) / cover_area
    if ratio < 0.005 or ratio > 0.4:
        return result
    smooth = cv2.GaussianBlur(cleaned, (3, 3), 0).astype(np.float32)[..., None] / 255.0
    return np.clip(result * (1.0 - smooth) + frame * smooth, 0, 255).astype(np.uint8)


def process_image(
    scene: np.ndarray,
    cover: np.ndarray,
    replacement: np.ndarray,
    *,
    preserve_occlusion: bool = False,
    min_matches: int = 10,
) -> tuple[np.ndarray, Detection | None]:
    match = locate_cover(scene, cover, min_matches=min_matches)
    if match is None:
        return scene.copy(), None
    result = replace_cover(scene, cover, replacement, match,
                           preserve_occlusion=preserve_occlusion)
    return result, match


def process_video(
    input_path: str | Path,
    cover: np.ndarray,
    overlay_video_path: str | Path,
    output_path: str | Path,
    *,
    max_frames: int | None = None,
) -> tuple[int, int]:
    """Process every frame; loop the overlay video. Outputs silent video (no audio)."""
    capture = cv2.VideoCapture(str(input_path))
    trailer = cv2.VideoCapture(str(overlay_video_path))
    if not capture.isOpened() or not trailer.isOpened():
        capture.release()
        trailer.release()
        raise RuntimeError("Could not open input video or replacement video")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        trailer.release()
        raise RuntimeError("Could not determine input video dimensions")
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (width, height))
    if not out.isOpened():
        capture.release()
        trailer.release()
        raise RuntimeError(f"Could not create output video: {output_path}")
    detector = cv2.SIFT_create()
    frame_count = detections = 0
    try:
        while max_frames is None or frame_count < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            match = locate_cover(frame, cover, min_matches=8, sift=detector)
            if match is not None:
                ok_trailer, replacement = trailer.read()
                if not ok_trailer:
                    trailer.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok_trailer, replacement = trailer.read()
                if ok_trailer:
                    frame = replace_cover(frame, cover, replacement, match)
                detections += 1
            out.write(frame)
            frame_count += 1
    finally:
        capture.release()
        trailer.release()
        out.release()
    return frame_count, detections
