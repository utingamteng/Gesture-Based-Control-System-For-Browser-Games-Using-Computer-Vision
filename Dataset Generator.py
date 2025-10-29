import cv2 as cv
import os
import argparse
from tqdm import tqdm

def extract_per_second(
    src_path: str,
    out_dir: str,
    interval_s: float = 1.0,
    start_s: float = 0.0,
    end_s: float | None = None,
    jpeg_quality: int = 95,
    prefix: str = "frame"
):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src_path}")

    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv.CAP_PROP_FPS) or 30
    if end_s is None:
        end_s = frame_count / fps if frame_count > 0 else 1e12

    total_frames = int((end_s - start_s) / interval_s)

    cap.set(cv.CAP_PROP_POS_MSEC, start_s * 1000.0)
    next_save_ms = start_s * 1000.0
    saved = 0

    with tqdm(total=total_frames, desc="Extracting frames", unit="img") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            pos_ms = cap.get(cv.CAP_PROP_POS_MSEC)
            if pos_ms > end_s * 1000.0:
                break

            if pos_ms + 0.0001 >= next_save_ms:
                fname = f"{prefix}_{saved:06d}.jpg"
                out_path = os.path.join(out_dir, fname)
                frame = cv.resize(frame, (1280, 720))
                cv.imwrite(out_path, frame, [cv.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
                saved += 1
                pbar.update(1)
                next_save_ms += interval_s * 1000.0

    cap.release()
    print(f"\nSaved {saved} frames to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract frames every N seconds.")
    ap.add_argument("src", help="Path to input video (e.g., input.mp4)")
    ap.add_argument("out", help="Output directory for extracted frames")
    ap.add_argument("--interval", type=float, default=1.0, help="Seconds between frames (default: 1.0)")
    ap.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0)")
    ap.add_argument("--end", type=float, default=None, help="End time in seconds (default: video end)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality 1–100 (default: 95)")
    ap.add_argument("--prefix", type=str, default="frame", help="Output filename prefix (default: frame)")
    args = ap.parse_args()

    extract_per_second(
        src_path=args.src,
        out_dir=args.out,
        interval_s=args.interval,
        start_s=args.start,
        end_s=args.end,
        jpeg_quality=args.quality,
        prefix=args.prefix
    )
