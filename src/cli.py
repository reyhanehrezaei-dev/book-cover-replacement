"""CLI for the reference-cover-to-image/video replacement pipeline."""
import argparse
from pathlib import Path

import cv2

from .cover_replacer import process_image, process_video, read_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace a known book cover using SIFT and homography.")
    subs = parser.add_subparsers(dest="command", required=True)

    still = subs.add_parser("image", help="Overlay a poster/image onto a detected cover in a photo")
    still.add_argument("--scene", required=True, help="Input scene/photo")
    still.add_argument("--cover", required=True, help="Reference book cover image")
    still.add_argument("--poster", required=True, help="Replacement image")
    still.add_argument("--output", required=True, help="Output PNG/JPG path")
    still.add_argument("--preserve-occlusion", action="store_true",
                       help="Experimental: preserve foreground objects via pixel differences")

    video = subs.add_parser("video", help="Overlay video frames onto a detected cover in a video")
    video.add_argument("--input", required=True, help="Input scene video")
    video.add_argument("--cover", required=True, help="Reference book cover image")
    video.add_argument("--overlay-video", required=True, help="Replacement content video")
    video.add_argument("--output", required=True, help="Output MP4 path")
    video.add_argument("--max-frames", type=int, default=None)

    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "image":
        scene = read_image(args.scene)
        cover = read_image(args.cover)
        poster = read_image(args.poster)
        result, match = process_image(scene, cover, poster,
                                      preserve_occlusion=args.preserve_occlusion)
        if match is None:
            raise SystemExit("Cover was not confidently detected; no output was saved.")
        if not cv2.imwrite(str(output), result):
            raise SystemExit(f"Could not write output: {output}")
        print(f"Saved {output} | matches={match.good_matches} inliers={match.inliers} "
              f"area_fraction={match.area_fraction:.3f}")
    else:
        cover = read_image(args.cover)
        count, matched = process_video(args.input, cover, args.overlay_video,
                                       output, max_frames=args.max_frames)
        print(f"Saved {output} | frames={count} detected_frames={matched} (no audio)")


if __name__ == "__main__":
    main()
