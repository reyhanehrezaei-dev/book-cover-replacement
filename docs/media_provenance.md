# Original Experiment Media — Provenance and Rights Notes

This document distinguishes original experiments from a separate synthetic reproducibility demo.

## Original sources

The user supplied `computer vision.zip` with:

- `4031390059/vision_part1.ipynb`: original image-processing code and saved outputs.
- `4031390059/vision_part2.ipynb`: original video-processing code and saved outputs.
- `4031390059/Result for part2/`: five full-length original experiment output videos.

**Notebook images**: all 14 PNG files in `examples/original/images/` are exact saved PNG output images extracted from those notebooks; they have not been regenerated, synthesized, or retouched for this repository. Thirteen correspond to the image notebook; one corresponds to a saved montage from the video notebook.

**Videos**: all five MP4 files in `examples/original/videos/` preserve the complete timeline of the original exported result videos, but were re-encoded to H.264 at 540 × 960 resolution to keep the GitHub repository small. Their source files were 1080 × 1920, 60 fps and approximately 11–14 seconds long. Neither the source export videos nor the converted files have audio. This re-encoding changes pixels/quality and file size; the underlying experiment footage is unchanged. The originally exported full-resolution files remain in the author's private project archive.

**Preview media**: the JPG files in `examples/original/previews/` are frames captured from the real exported results; `real_video_sample.gif` is a short animated excerpt derived from the real Towers result. The GIF's reduced frame rate and size are for preview only.

## Image filenames and associations

| Detection image | Matching notebook replacement image |
|---|---|
| `fellowship_1_detection.png` | Not present as a saved final output in supplied notebook |
| `fellowship_2_detection.png` | `fellowship_2_result.png` |
| `hobbit_1_detection.png` | `hobbit_1_result.png` |
| `hobbit_2_detection.png` | `hobbit_2_result.png` |
| `hobbit_3_detection.png` | `hobbit_3_result.png` |
| `lotr_return_detection.png` | `lotr_return_result.png` |
| `lotr_towers_detection.png` | `lotr_towers_result.png` |

`video_notebook_frame_montage.png` is a saved notebook output from the video processing experiments.

## Publication and copyright reminder

The author stated that she took the project photographs and produced the video composites. That establishes provenance of her practical work, **not necessarily rights to underlying third-party book art, movie posters, trailer frames, music, or other protected works** appearing within those composites. For public sharing, review permissions, the intended jurisdiction's copyright exceptions, and the source licensors' terms. When permissions are uncertain, consider shorter excerpts, blurred/redacted underlying artwork, or permission-cleared replacements instead of publishing third-party video material in full.

The standalone synthetic media under `examples/synthetic/` were generated as algorithm smoke-test inputs and are not evidence of the original book/movie experiments.

The code repository intentionally does not include the original 200+ MB full-resolution exports or third-party source poster/trailer files.
