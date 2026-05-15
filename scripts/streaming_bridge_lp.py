"""LivePortrait streaming daemon: webcam RGB → FasterLivePortrait → v4l2.

Per-frame inference (no batching). FasterLivePortrait reports ~13 ms/frame
on RTX 5090 with the TRT engines from
``/home/newub/w/FasterLivePortrait/checkpoints/liveportrait_onnx/``, so
glass-to-OBS latency is dominated by camera capture (~33 ms at 30 fps) +
infer (~13 ms) + sink pipeline depth, NOT by cohort-batching gymnastics
the way the PersonaLive daemon was.

Usage (run inside the FasterLivePortrait venv — it has TRT + onnxruntime
GPU build wired up; our project venv does not):

    sudo modprobe v4l2loopback devices=1 video_nr=10 \\
        card_label=PersonaLive exclusive_caps=1

    PYTHONPATH=/home/newub/w/vamp-interface/src \\
    /home/newub/w/FasterLivePortrait/.venv/bin/python \\
        /home/newub/w/vamp-interface/scripts/streaming_bridge_lp.py \\
        --reference /home/newub/w/vamp-interface/data/anchors/photoreal_ffhq/east_asian__adult__m__41526046.png \\
        --cam_index 0 \\
        --device_path /dev/video10 \\
        --fps 25

For smoke-testing without v4l2loopback, pass ``--mp4_out path.mp4``.

Pipeline:

  cv2.VideoCapture(cam) -> (BGR ndarray)
        ↓
  FasterLivePortraitPipeline.run()           # ~13 ms TRT
        → (driver_crop, animated_512, paste_back, motion_info)
        ↓
  PacedSinkWriter(prebuffer=2)               # background drain @ fps
        ↓
  V4L2Sink (ffmpeg rgb24 → v4l2 yuv420p)     # OBS capture device

Why so much smaller than the PersonaLive daemon?

* No bridge: LivePortrait's MotionExtractor consumes the driver crop
  directly. We don't produce blendshapes, we don't run a student, we
  don't install seams. That whole class of artifacts is gone.
* No batching: ``pipe.run()`` is per-frame. We don't need cohort sampling
  windows or evenly-spaced packet draws.
* Almost no prebuffer: PersonaLive needed ``--writer_prebuffer 24`` to
  cover the 820 ms ``prepare_v2`` gap at every batch boundary. LP has
  no analogous gap; ``--writer_prebuffer 2`` (~80 ms) is enough.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

# --- repo + FLP imports ------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

FLP_ROOT = Path("/home/newub/w/FasterLivePortrait")
if str(FLP_ROOT) not in sys.path:
    sys.path.insert(0, str(FLP_ROOT))

import cv2  # noqa: E402  (after sys.path setup so FLP venv resolves)
from omegaconf import OmegaConf  # noqa: E402

from arkit_bridge.paced_sink import PacedSinkWriter  # noqa: E402
from arkit_bridge.v4l2_sink import V4L2Sink  # noqa: E402


def _load_pipeline(cfg_path: str, animal: bool, crop_driver: bool = False,
                   pasteback: bool = False, crop_overrides: dict | None = None,
                   no_relative_motion: bool = False):
    """Construct FasterLivePortraitPipeline. CWD-sensitive (model_path is
    relative inside the cfg yaml), so we chdir into FLP_ROOT for load."""
    from src.pipelines.faster_live_portrait_pipeline import (  # type: ignore
        FasterLivePortraitPipeline,
    )

    cfg = OmegaConf.load(cfg_path)
    # We always paste-back-OFF: we want the 512x512 animated crop only,
    # not a re-pasted full webcam frame. That matches what ``-crop.mp4``
    # produces in run.py and what OBS wants as a virtual camera.
    cfg.infer_params.flag_pasteback = bool(pasteback)
    cfg.infer_params.flag_crop_driving_video = bool(crop_driver)
    if no_relative_motion:
        cfg.infer_params.flag_relative_motion = False
    if crop_overrides:
        for k, v in crop_overrides.items():
            if v is not None:
                cfg.crop_params[k] = float(v)

    saved_cwd = os.getcwd()
    try:
        os.chdir(FLP_ROOT)
        pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=animal)
    finally:
        os.chdir(saved_cwd)
    return pipe


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True,
                    help="Source portrait image (the face that gets animated).")
    ap.add_argument("--cam_index", type=int, default=0)
    ap.add_argument("--driver_video",
                    help="If set, read driver frames from this mp4 instead "
                    "of the webcam (offline smoke test). Loops at EOF.")
    ap.add_argument("--device_path", default="/dev/video10",
                    help="v4l2loopback device, OR path.mp4 if --mp4_out is set.")
    ap.add_argument("--mp4_out",
                    help="If set, write to this mp4 instead of v4l2 (smoke test).")
    ap.add_argument("--fps", type=int, default=25,
                    help="Sink drain FPS. Set close to the camera's actual "
                    "rate (usually 30) for least repeat-stutter.")
    ap.add_argument("--writer_prebuffer", type=int, default=2,
                    help="Frames to accumulate before sink starts draining "
                    "(LP needs ~zero; 2 absorbs cv2/MP detection jitter).")
    ap.add_argument("--cfg",
                    default=str(FLP_ROOT / "configs" / "trt_infer.yaml"))
    ap.add_argument("--animal", action="store_true",
                    help="Use the Animals checkpoint (different MotionExtractor).")
    ap.add_argument("--crop_driver", action="store_true",
                    help="Enable FLP's internal face-crop on the driver "
                    "stream (retinaface + 106-landmark + dri_scale crop). "
                    "Default off matches trt_infer.yaml — webcam frame is "
                    "resized whole to 256². On gives properly framed "
                    "face input to MotionExtractor.")
    ap.add_argument("--no_relative_motion", action="store_true",
                    help="Disable LP relative-motion mode. Default ON: "
                    "driver scale/rotation/exp are taken as ratios vs "
                    "driver frame 0 (causes 'head pumping' on big "
                    "expressions). Off: absolute motion, expression "
                    "scale-pump goes away but pose interpretation shifts.")
    ap.add_argument("--scale_clamp", type=float, default=None,
                    help="Clamp MotionExtractor's per-frame scale to the "
                    "first driver frame's scale × (1 ± BAND). E.g. 0.03 → "
                    "[0.97, 1.03] of frame-0 scale. Tames 'head pumping' "
                    "on big expressions while preserving relative-motion "
                    "expression handling (no squinted/tight-lipped artifact "
                    "from absolute mode). Default: no clamp.")
    ap.add_argument("--smooth_motion", action="store_true",
                    help="Apply OneEuroFilter to MotionExtractor outputs "
                    "(pitch/yaw/roll/t/exp/scale) on every frame. FLP only "
                    "smooths these in is_source_video=True path; this "
                    "extends smoothing to image-source mode (our case). "
                    "Cuts per-frame jitter at the cost of a few ms of "
                    "lag on big motion onsets.")
    ap.add_argument("--pasteback", action="store_true",
                    help="Output = full source frame with the animated "
                    "head re-pasted into it (shoulders, background, etc). "
                    "Default off → output is just the 512² animated face "
                    "crop. Sink width/height auto-derive from source size.")
    # Crop knobs — override trt_infer.yaml crop_params defaults.
    # Source-side (controls how much of the SOURCE portrait is the canvas):
    ap.add_argument("--src_scale", type=float, default=None,
                    help="Source-crop scale around the face (default 2.3). "
                    "Larger → wider FOV around the face, includes more "
                    "shoulders/background; smaller → tighter head crop.")
    ap.add_argument("--src_vx_ratio", type=float, default=None,
                    help="Source-crop horizontal offset, frac of crop size.")
    ap.add_argument("--src_vy_ratio", type=float, default=None,
                    help="Source-crop vertical offset (default -0.125 = up).")
    # Driver-side (only used when --crop_driver is set):
    ap.add_argument("--dri_scale", type=float, default=None,
                    help="Driver-crop scale (default 2.2; needs --crop_driver).")
    ap.add_argument("--dri_vx_ratio", type=float, default=None)
    ap.add_argument("--dri_vy_ratio", type=float, default=None,
                    help="Driver-crop vertical offset (default -0.1).")
    ap.add_argument("--log_every_s", type=float, default=2.0)
    ap.add_argument("--max_seconds", type=float, default=0.0,
                    help="If >0, exit after this many seconds (smoke test).")
    args = ap.parse_args()

    # --- pipeline -----------------------------------------------------
    print(f"[lp] loading FLP pipeline (animal={args.animal}) ...", flush=True)
    crop_overrides = {
        "src_scale": args.src_scale,
        "src_vx_ratio": args.src_vx_ratio,
        "src_vy_ratio": args.src_vy_ratio,
        "dri_scale": args.dri_scale,
        "dri_vx_ratio": args.dri_vx_ratio,
        "dri_vy_ratio": args.dri_vy_ratio,
    }
    pipe = _load_pipeline(args.cfg, animal=args.animal,
                          crop_driver=args.crop_driver,
                          pasteback=args.pasteback,
                          crop_overrides=crop_overrides,
                          no_relative_motion=args.no_relative_motion)
    print(f"[lp] crop_overrides: "
          f"{ {k: v for k, v in crop_overrides.items() if v is not None} }",
          flush=True)
    print(f"[lp] preparing source: {args.reference}", flush=True)
    saved_cwd = os.getcwd()
    try:
        os.chdir(FLP_ROOT)  # prepare_source resolves relative paths internally
        ok = pipe.prepare_source(args.reference, realtime=False)
    finally:
        os.chdir(saved_cwd)
    if not ok:
        print(f"[lp] no face in source {args.reference}; abort.")
        sys.exit(1)
    src_img = pipe.src_imgs[0]
    src_info = pipe.src_infos[0]

    # --- driver-side MotionExtractor smoothing / scale clamp ----------
    # Applied only on the DRIVER stream — prepare_source already ran one
    # pass through motion_extractor on the SOURCE image, and we don't
    # want to clamp/smooth that one. Patching now means only subsequent
    # (per-frame, driver) calls are affected.
    if args.smooth_motion or args.scale_clamp is not None:
        from src.utils.utils import OneEuroFilter  # type: ignore
        me = pipe.model_dict["motion_extractor"]
        _orig_predict = me.predict

        # Per-stream filters (created lazily on the first driver frame).
        # Tuned per FLP's existing R_d_smooth defaults: mincutoff=4, beta=0.3.
        _state: dict = {
            "scale_0": None,
            "filters": None,  # (pitch, yaw, roll, t, exp, scale)
            "frame_id": 0,
        }

        def _patched_predict(img_crop):
            (pitch, yaw, roll, t, exp, scale, kp) = _orig_predict(img_crop)

            if args.smooth_motion:
                if _state["filters"] is None:
                    _state["filters"] = tuple(
                        OneEuroFilter(mincutoff=4.0, beta=0.3) for _ in range(6)
                    )
                f_p, f_y, f_r, f_t, f_e, f_s = _state["filters"]
                pitch = f_p.process(pitch)
                yaw = f_y.process(yaw)
                roll = f_r.process(roll)
                t = f_t.process(t)
                exp = f_e.process(exp)
                scale = f_s.process(scale)

            if args.scale_clamp is not None:
                if _state["scale_0"] is None:
                    _state["scale_0"] = float(np.asarray(scale).flatten()[0])
                s0 = _state["scale_0"]
                lo = s0 * (1.0 - args.scale_clamp)
                hi = s0 * (1.0 + args.scale_clamp)
                scale = np.clip(scale, lo, hi)

            _state["frame_id"] += 1
            return pitch, yaw, roll, t, exp, scale, kp

        me.predict = _patched_predict
        smooth_str = "ON" if args.smooth_motion else "off"
        clamp_str = (f"±{args.scale_clamp:.2f}"
                     if args.scale_clamp is not None else "off")
        print(f"[lp] driver-side MotionExtractor patches: "
              f"smooth={smooth_str}  scale_clamp={clamp_str}", flush=True)

    # --- sink ---------------------------------------------------------
    # Pasteback writes the full source frame; head-only writes the 512² crop.
    if args.pasteback:
        sink_h, sink_w = src_img.shape[:2]
        print(f"[lp] pasteback ON; sink dims = {sink_w}x{sink_h} "
              f"(source-frame size)", flush=True)
    else:
        sink_w, sink_h = 512, 512
    if args.mp4_out:
        sink = V4L2Sink(device=args.mp4_out, fps=args.fps,
                        width=sink_w, height=sink_h,
                        output_format="mp4")
    else:
        sink = V4L2Sink(device=args.device_path, fps=args.fps,
                        width=sink_w, height=sink_h,
                        output_format="v4l2")
    sink.open()
    writer = PacedSinkWriter(sink, fps=args.fps, max_queue=128,
                             prebuffer=args.writer_prebuffer)
    writer.start()

    # --- camera -------------------------------------------------------
    if args.driver_video:
        cap = cv2.VideoCapture(args.driver_video)
        if not cap.isOpened():
            print(f"[lp] driver mp4 open failed: {args.driver_video}")
            writer.stop(); sink.close()
            sys.exit(1)
        offline = True
    else:
        cap = cv2.VideoCapture(args.cam_index)
        if not cap.isOpened():
            print(f"[lp] cv2.VideoCapture({args.cam_index}) failed.")
            writer.stop(); sink.close()
            sys.exit(1)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always read freshest
        offline = False

    # --- run ----------------------------------------------------------
    stop_flag = {"stop": False}

    def _sigint(*_):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    print(f"[lp] running. fps={args.fps} prebuffer={args.writer_prebuffer} "
          f"sink={args.mp4_out or args.device_path}", flush=True)

    t_start = time.time()
    t_last_log = t_start
    frame_id = 0
    infer_times: list[float] = []
    no_face_count = 0

    try:
        while not stop_flag["stop"]:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                if offline:
                    # Loop at EOF for the offline driver — keeps the smoke
                    # output streaming until --max_seconds elapses.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.005)
                continue

            t0 = time.time()
            try:
                _dri_crop, out_crop, out_org, _motion = pipe.run(
                    bgr, src_img, src_info,
                    first_frame=(frame_id == 0),
                )
            except Exception as e:
                print(f"[lp] pipe.run() raised: {e!r}", flush=True)
                continue
            t_infer = time.time() - t0

            if out_crop is None:
                # No face detected this frame — repeat-last in writer is
                # the right fallback. Don't push anything.
                no_face_count += 1
                continue

            infer_times.append(t_infer)
            frame_id += 1
            # FLP returns RGB 512x512 (out_crop) and the full pastebacked
            # source frame (out_org) at original source dims when
            # flag_pasteback is set. Pick the right one for the sink.
            frame_out = out_org if args.pasteback else out_crop
            if frame_out is None:
                # Pasteback can fail to compose if stitching/crop bookkeeping
                # didn't run (e.g. no_face on first frame). Fall back to crop.
                frame_out = out_crop
            writer.push([np.ascontiguousarray(frame_out)])

            now = time.time()
            if now - t_last_log >= args.log_every_s:
                snap = writer.snapshot()
                window = max(now - t_last_log, 1e-6)
                infer_med_ms = float(np.median(infer_times)) * 1000 if infer_times else 0.0
                infer_p95_ms = float(np.percentile(infer_times, 95)) * 1000 if infer_times else 0.0
                print(
                    f"[lp t={now - t_start:6.1f}s] "
                    f"frames={frame_id} no_face={no_face_count} "
                    f"infer_med={infer_med_ms:5.1f}ms p95={infer_p95_ms:5.1f}ms | "
                    f"sink writes={snap['writes']} fresh={snap['fresh']} "
                    f"repeat={snap['repeat']} dropped={snap['dropped']} "
                    f"q_mean={snap['depth_mean']:.1f} q_max={snap['depth_max']} | "
                    f"write_fps={snap['writes'] / window:.1f} "
                    f"fresh_fps={snap['fresh'] / window:.1f}",
                    flush=True,
                )
                writer.reset_window()
                infer_times.clear()
                t_last_log = now

            if args.max_seconds and (now - t_start) >= args.max_seconds:
                break
    finally:
        try:
            cap.release()
        except Exception:
            pass
        writer.stop()
        sink.close()
        print(f"[lp] stopped. total_frames={frame_id} no_face={no_face_count}",
              flush=True)


if __name__ == "__main__":
    main()
