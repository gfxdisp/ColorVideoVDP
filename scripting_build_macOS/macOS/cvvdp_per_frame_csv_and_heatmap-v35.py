#!/opt/homebrew/anaconda3/envs/cvvdp/bin/python
"""
cvvdp_per_frame_csv_and_heatmap_v35.py

Usage

usage: cvvdp_per_frame_csv_and_heatmap_subsampOPT-v35.py [-h] [--mode {supra-threshold,threshold,raw}] [--display DISPLAY]
                                                         [--pix-per-deg PIX_PER_DEG] [--temp-window TEMP_WINDOW] [--device DEVICE]
                                                         [--display-res DISPLAY_RES]
                                                         [--chroma-filter {point,bilinear,bicubic,spline16,spline36,lanczos}] [--no-compare]
                                                         [--keep-work] [--limit-frames LIMIT_FRAMES] [--verbose] [--temp-resample]
                                                         [--temp-padding {replicate,pingpong,circular}]
                                                         [--dump-channels {temporal,lpyr,difference} [{temporal,lpyr,difference} ...]]
                                                         [--dump-output-dir DUMP_OUTPUT_DIR]
                                                         [ref] [test] [outdir]

Modes
-----
--mode supra-threshold | threshold | raw
  supra-threshold : perceptual supra-threshold difference map (often most useful)
  threshold       : detection-threshold map
  raw             : raw perceptual error energy map (implementation-dependent)

Example:
  python cvvdp_per_frame_csv_and_heatmap_v35.py ref.mov test.mov out \
    --display NBCU_65inch_hdr_pq_2knit --device mps --mode supra-threshold \
    --temp-window 0

Two analysis modes written to CSV
---------------------------------
jod_total
    Native reference vs test.
    This measures real delivered quality (444 reference vs encoded test).

jod_total_ref420sim
    Simulated-420 reference vs test.
    This reduces bias from chroma subsampling by putting the reference through
    444/422 -> 420 -> 444 reconstruction before comparison.
    
Features
--------
• Drag-and-drop friendly (interactive prompts if paths not supplied)
• Path cleanup for macOS Terminal drag-drop (quotes, escaped spaces, CR)
• Extract REF+TEST to 16-bit RGB TIFF sequences ONCE (frame index source of truth)
• REF is extracted in TWO ways:
  1) native full-resolution RGB reference
  2) simulated-4:2:0 reference (444/422 -> 420 -> 444) for encoder-only fairness analysis
• TIFF extraction uses zscale by default with filter=spline36
• Per-frame CVVDP JOD + per-frame heatmap PNG (one PNG per source frame)
  - Heatmaps are generated from the NATIVE reference analysis
• Optional temporal window: run cvvdp on a multi-frame window (i-k):(i+k)
• Optional cvvdp debug dumps via:
    --dump-channels temporal|lpyr|difference [...]
  - Can optionally preserve those outputs with:
    --dump-output-dir /path/to/folder
• Heatmap PNG sequence → ProRes MOV (CFR, no frame blending)
• Optional side-by-side compare MOV: (TEST | HEATMAP MOV)
  - Automatically scales TEST to match HEATMAP height so hstack never fails
  - If TEST/source is PQ, compare MOV is automatically mastered/tagged as PQ/BT.2020
    and the heatmap leg is up-mapped from SDR/BT.709 → linear display light → 2x scale
    → BT.2020 → PQ so it looks correct in the HDR compare output
• Automatic PNG cleanup after successful heatmap MOV creation (keeps PNGs if MOV fails)
• CSV includes legends + options used (# key=value lines)

"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v"}


# -------------------------------------------------------------
# Path cleaning (drag-drop safe)
# -------------------------------------------------------------

def clean_path(p: str) -> str:
    p = p.strip().replace("\r", "")
    if len(p) >= 2 and ((p[0] == '"' and p[-1] == '"') or (p[0] == "'" and p[-1] == "'")):
        p = p[1:-1]
    p = p.replace("\\ ", " ")
    return p


def prompt_path(prompt: str) -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    buf: List[str] = []
    while True:
        ch = sys.stdin.read(1)
        if ch == "":
            raise RuntimeError("EOF while reading input.")
        if ch in ("\n", "\r"):
            line = clean_path("".join(buf))
            if line:
                return line
            sys.stdout.write(prompt)
            sys.stdout.flush()
            buf = []
            continue
        buf.append(ch)


# -------------------------------------------------------------
# Command runner / exec discovery
# -------------------------------------------------------------

def run(cmd: List[str], verbose: bool = False) -> str:
    if verbose:
        print(" ".join(str(c) for c in cmd))
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(map(str, cmd))}\n\n{p.stdout}")
    return p.stdout


def which_or_raise(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise RuntimeError(f"Required executable not found on PATH: {name}")
    return p


def find_cvvdp_exe() -> str:
    p = shutil.which("cvvdp")
    if p:
        return p

    pybin = Path(sys.executable).resolve().parent
    cand = pybin / "cvvdp"
    if cand.exists():
        return str(cand)

    conda_prefix = Path(sys.executable).resolve().parent.parent
    cand2 = conda_prefix / "bin" / "cvvdp"
    if cand2.exists():
        return str(cand2)

    candidates = [
        Path("/opt/homebrew/anaconda3/envs"),
        Path("/opt/homebrew/Caskroom/miniconda/base/envs"),
        Path.home() / "miniconda3" / "envs",
        Path.home() / "anaconda3" / "envs",
        Path("/usr/local/anaconda3/envs"),
        Path("/usr/local/miniconda3/envs"),
    ]
    for root in candidates:
        if root.exists():
            for exe in root.glob("*/bin/cvvdp"):
                if exe.exists():
                    return str(exe)

    raise RuntimeError(
        "Could not locate 'cvvdp' executable.\n"
        f"sys.executable = {sys.executable}\n"
        "Fix options:\n"
        "  A) Run with env python, e.g.: /opt/homebrew/anaconda3/envs/cvvdp/bin/python script.py\n"
        "  B) Put env on PATH, e.g.: export PATH=/opt/homebrew/anaconda3/envs/cvvdp/bin:$PATH\n"
        "  C) Or hardcode CVVDP_EXE in this script.\n"
    )


# -------------------------------------------------------------
# ffprobe helpers
# -------------------------------------------------------------

def ffprobe_stream_meta(path: str) -> Dict[str, str]:
    ffprobe = which_or_raise("ffprobe")
    out = run([
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,color_space,color_transfer,color_primaries,bit_rate,width,height,r_frame_rate,color_range",
        "-of", "json",
        path
    ])
    j = json.loads(out)
    s = (j.get("streams") or [{}])[0]

    def g(k: str) -> str:
        v = s.get(k, "")
        return "" if v is None else str(v)

    return {
        "codec_name": g("codec_name"),
        "pix_fmt": g("pix_fmt"),
        "color_space": g("color_space"),
        "color_transfer": g("color_transfer"),
        "color_primaries": g("color_primaries"),
        "color_range": g("color_range"),
        "bit_rate": g("bit_rate"),
        "width": g("width"),
        "height": g("height"),
        "r_frame_rate": g("r_frame_rate"),
    }


def ffprobe_r_frame_rate(path: str) -> str:
    r = ffprobe_stream_meta(path).get("r_frame_rate", "")
    if not r:
        raise RuntimeError("ffprobe did not return r_frame_rate.")
    return r


def r_frame_rate_to_float(r: str) -> float:
    if "/" in r:
        n, d = r.split("/")
        return float(n) / float(d)
    return float(r)


def ffprobe_wh(path: str) -> Tuple[int, int]:
    ffprobe = which_or_raise("ffprobe")
    out = run([
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        path
    ])
    s = out.strip().splitlines()[0]
    w, h = s.split("x")
    return int(w), int(h)


def is_pq_video(meta: Dict[str, str]) -> bool:
    trc = (meta.get("color_transfer") or "").strip().lower()
    return trc in {"smpte2084", "pq"}


def ffmpeg_has_zscale() -> bool:
    ffmpeg = which_or_raise("ffmpeg")
    try:
        out = run([ffmpeg, "-hide_banner", "-filters"])
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "zscale":
                return True
        return False
    except Exception:
        return False


# -------------------------------------------------------------
# Display-res parsing
# -------------------------------------------------------------

def parse_display_res(res: str) -> Tuple[int, int]:
    s = res.lower().replace(" ", "")
    if "x" not in s:
        raise ValueError(f"--display-res must be like 3840x2160, got: {res}")
    w, h = s.split("x", 1)
    return int(w), int(h)


# -------------------------------------------------------------
# Frame extraction (TIFF)
# -------------------------------------------------------------

def extract_tiffs_native(video: str,
                         out_dir: Path,
                         prefix: str,
                         display_res: Tuple[int, int],
                         chroma_filter: str,
                         verbose: bool = False) -> Path:
    """
    Extract frames as RGB48 TIFF using zscale for resize, chroma reconstruction,
    and color conversion.
    """
    ffmpeg = which_or_raise("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{prefix}_%06d.tif"

    w, h = display_res
    vf = (
        f"zscale="
        f"w={w}:h={h}:"
        f"filter={chroma_filter}:"
        f"dither=error_diffusion,"
        f"format=rgb48le"
    )

    run([
        ffmpeg, "-hide_banner", "-y",
        "-i", video,
        "-vsync", "0",
        "-start_number", "0",
        "-vf", vf,
        str(pattern)
    ], verbose=verbose)

    return pattern


def extract_tiffs_ref420sim(video: str,
                            out_dir: Path,
                            prefix: str,
                            display_res: Tuple[int, int],
                            chroma_filter: str,
                            verbose: bool = False) -> Path:
    """
    Extract reference frames after simulating 4:2:0:
      source -> scale/convert -> yuv420p10le -> upsample -> rgb48le TIFF

    This is useful for encoder-only fairness analysis when the true reference is 444/422.
    """
    ffmpeg = which_or_raise("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{prefix}_%06d.tif"

    w, h = display_res
    vf = (
        f"zscale="
        f"w={w}:h={h}:"
        f"filter={chroma_filter}:"
        f"dither=error_diffusion,"
        f"format=yuv420p10le,"
        f"zscale="
        f"w={w}:h={h}:"
        f"filter={chroma_filter}:"
        f"dither=error_diffusion,"
        f"format=rgb48le"
    )

    run([
        ffmpeg, "-hide_banner", "-y",
        "-i", video,
        "-vsync", "0",
        "-start_number", "0",
        "-vf", vf,
        str(pattern)
    ], verbose=verbose)

    return pattern


def count_tiffs(out_dir: Path, prefix: str) -> int:
    return len(list(out_dir.glob(f"{prefix}_*.tif")))


# -------------------------------------------------------------
# CVVDP parsing helpers
# -------------------------------------------------------------

def parse_metric_value(output: str) -> Optional[float]:
    s = output.strip()
    if not s:
        return None

    if "\n" not in s:
        try:
            return float(s)
        except Exception:
            pass

    for line in s.splitlines():
        t = line.strip()
        if "[JOD]" in t and "=" in t:
            try:
                return float(t.split("=", 1)[1].split()[0])
            except Exception:
                continue

    for line in s.splitlines():
        t = line.strip()
        if "=" in t:
            try:
                return float(t.split("=", 1)[1].split()[0])
            except Exception:
                pass
        else:
            try:
                return float(t.split()[0])
            except Exception:
                pass

    return None


# -------------------------------------------------------------
# Heatmap extraction helpers
# -------------------------------------------------------------

def _extract_center_frame_from_video(src_video: Path, out_png: Path, center_index: int, verbose: bool = False) -> None:
    ffmpeg = which_or_raise("ffmpeg")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    vf = f"select='eq(n\\,{center_index})'"
    run([
        ffmpeg, "-hide_banner", "-y",
        "-i", str(src_video),
        "-vf", vf,
        "-frames:v", "1",
        str(out_png)
    ], verbose=verbose)


def _extract_single_png_or_convert(src_img: Path, out_png: Path, verbose: bool = False) -> None:
    ffmpeg = which_or_raise("ffmpeg")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if src_img.suffix.lower() == ".png":
        out_png.write_bytes(src_img.read_bytes())
    else:
        run([
            ffmpeg, "-hide_banner", "-y",
            "-i", str(src_img),
            "-frames:v", "1",
            str(out_png)
        ], verbose=verbose)


# -------------------------------------------------------------
# Run CVVDP: JOD + HEATMAP (window) → center-frame heatmap PNG
# -------------------------------------------------------------

def run_cvvdp_window_and_heatmap(
    cvvdp_exe: str,
    ref_pat: Path,
    test_pat: Path,
    center_frame: int,
    temp_window: int,
    display: str,
    device: str,
    fps: float,
    heatmap_mode: str,
    pix_per_deg: Optional[float],
    temp_resample: bool,
    temp_padding: str,
    dump_channels: List[str],
    dump_output_dir: str,
    out_png: Path,
    verbose: bool = False
) -> float:
    k = max(0, int(temp_window))
    start = max(0, int(center_frame) - k)
    end = int(center_frame) + k

    temp_ctx = None
    try:
        if dump_channels and dump_output_dir:
            td_path = Path(dump_output_dir).expanduser().resolve() / f"frame_{center_frame:06d}"
            if td_path.exists():
                shutil.rmtree(td_path)
            td_path.mkdir(parents=True, exist_ok=True)
        else:
            temp_ctx = tempfile.TemporaryDirectory(prefix="cvvdp_win_")
            td_path = Path(temp_ctx.name)

        cmd = [
            str(cvvdp_exe),
            "--ffmpeg-cc",
            "--device", str(device),
            "--display", str(display),
            "--fps", f"{float(fps):.12f}",
            "--frames", f"{start}:{end}",
            "--heatmap", str(heatmap_mode),
            "--output-dir", str(td_path),
            "--ref", str(ref_pat),
            "--test", str(test_pat),
        ]

        if temp_resample:
            cmd.insert(1, "--temp-resample")

        if temp_padding:
            cmd.insert(1, temp_padding)
            cmd.insert(1, "--temp-padding")

        if dump_channels:
            idx = cmd.index("--output-dir")
            cmd[idx:idx] = ["--dump-channels", *dump_channels]

        if pix_per_deg is not None:
            idx = cmd.index("--fps")
            cmd[idx:idx] = ["--pix-per-deg", str(pix_per_deg)]

        out = run(cmd, verbose=verbose)

        jod = parse_metric_value(out)
        if jod is None:
            raise RuntimeError(
                f"Could not parse JOD for center_frame={center_frame}\n--- cvvdp output ---\n{out}"
            )

        heat_imgs = sorted([
            p for p in td_path.iterdir()
            if p.is_file()
            and "heatmap" in p.name.lower()
            and p.suffix.lower() in {".png", ".tif", ".tiff"}
        ])
        heat_vids = sorted([
            p for p in td_path.iterdir()
            if p.is_file()
            and "heatmap" in p.name.lower()
            and p.suffix.lower() in VIDEO_EXTS
        ])

        if heat_imgs:
            _extract_single_png_or_convert(heat_imgs[0], out_png, verbose=verbose)
        elif heat_vids:
            center_idx = max(0, int(center_frame) - start)
            _extract_center_frame_from_video(heat_vids[0], out_png, center_idx, verbose=verbose)
        else:
            listing = "\n".join([p.name for p in td_path.iterdir()])
            raise RuntimeError(
                f"No heatmap output produced for center_frame={center_frame}.\nDir:\n{listing}"
            )

        return float(jod)

    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


def run_cvvdp_jod_only(
    cvvdp_exe: str,
    ref_pat: Path,
    test_pat: Path,
    center_frame: int,
    temp_window: int,
    display: str,
    device: str,
    fps: float,
    pix_per_deg: Optional[float],
    temp_resample: bool,
    temp_padding: str,
    verbose: bool = False
) -> float:
    k = max(0, int(temp_window))
    start = max(0, int(center_frame) - k)
    end = int(center_frame) + k

    cmd = [
        str(cvvdp_exe),
        "-q",
        "--ffmpeg-cc",
        "--device", str(device),
        "--display", str(display),
        "--fps", f"{float(fps):.12f}",
        "--frames", f"{start}:{end}",
        "--ref", str(ref_pat),
        "--test", str(test_pat),
    ]

    if temp_resample:
        cmd.insert(2, "--temp-resample")

    if temp_padding:
        cmd.insert(2, temp_padding)
        cmd.insert(2, "--temp-padding")

    if pix_per_deg is not None:
        idx = cmd.index("--fps")
        cmd[idx:idx] = ["--pix-per-deg", str(pix_per_deg)]

    out = run(cmd, verbose=verbose)
    v = parse_metric_value(out)
    if v is None:
        raise RuntimeError(f"Could not parse JOD for frame={center_frame}\n--- output ---\n{out}")
    return float(v)


# -------------------------------------------------------------
# Heatmap MOV + Compare MOV
# -------------------------------------------------------------

def encode_png_sequence_to_mov(
    png_pattern: str,
    fps_ffmpeg: str,
    out_mov: Path,
    source_is_pq: bool,
    verbose: bool = False
) -> None:
    ffmpeg = which_or_raise("ffmpeg")

    if source_is_pq:
        vf = (
            "zscale="
            "matrixin=gbr:"
            "transferin=bt709:"
            "primariesin=bt709:"
            "rangein=full:"
            "matrix=gbr:"
            "transfer=linear:"
            "primaries=bt2020:"
            "range=full,"
            "lutrgb="
            "r='clip(val*2,0,maxval)':"
            "g='clip(val*2,0,maxval)':"
            "b='clip(val*2,0,maxval)',"
            "zscale="
            "matrixin=gbr:"
            "transferin=linear:"
            "primariesin=bt2020:"
            "rangein=full:"
            "matrix=gbr:"
            "transfer=smpte2084:"
            "primaries=bt2020:"
            "range=full,"
            "zscale="
            "matrixin=gbr:"
            "transferin=smpte2084:"
            "primariesin=bt2020:"
            "rangein=full:"
            "matrix=bt2020nc:"
            "transfer=smpte2084:"
            "primaries=bt2020:"
            "range=tv,"
            "setparams="
            "color_primaries=bt2020:"
            "color_trc=smpte2084:"
            "colorspace=bt2020nc:"
            "range=tv"
        )

        run([
            ffmpeg, "-hide_banner", "-y",
            "-framerate", fps_ffmpeg,
            "-i", png_pattern,
            "-vf", vf,
            "-fps_mode", "cfr",
            "-c:v", "prores_ks", "-profile:v", "3", "-q:v", "7",
            "-pix_fmt", "yuv422p10le",
            "-movflags", "+write_colr",
            "-color_primaries", "bt2020",
            "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc",
            str(out_mov)
        ], verbose=verbose)

    else:
        run([
            ffmpeg, "-hide_banner", "-y",
            "-framerate", fps_ffmpeg,
            "-i", png_pattern,
            "-fps_mode", "cfr",
            "-c:v", "prores_ks", "-profile:v", "3", "-q:v", "7",
            "-pix_fmt", "yuv422p10le",
            str(out_mov)
        ], verbose=verbose)


def encode_compare_mov(
    test: str,
    heat_mov: str,
    out_compare_mov: Path,
    fps_ffmpeg: str,
    test_meta: Dict[str, str],
    verbose: bool = False
) -> None:
    ffmpeg = which_or_raise("ffmpeg")
    _, heat_h = ffprobe_wh(heat_mov)

    pq_mode = is_pq_video(test_meta)

    if pq_mode:
        filt = (
            f"[0:v]"
            f"setpts=PTS-STARTPTS,"
            f"fps=fps={fps_ffmpeg}:round=near,"
            f"scale=-2:{heat_h}:flags=bicubic,"
            f"setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc:range=tv"
            f"[v0];"
            f"[1:v]"
            f"setpts=PTS-STARTPTS,"
            f"fps=fps={fps_ffmpeg}:round=near,"
            f"setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc:range=tv"
            f"[v1];"
            f"[v0][v1]hstack=inputs=2,"
            f"setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc:range=tv"
            f"[v]"
        )

        run([
            ffmpeg, "-hide_banner", "-y",
            "-i", test,
            "-i", heat_mov,
            "-filter_complex", filt,
            "-map", "[v]",
            "-an",
            "-fps_mode", "cfr",
            "-c:v", "prores_ks", "-profile:v", "3", "-q:v", "7",
            "-pix_fmt", "yuv422p10le",
            "-movflags", "+write_colr",
            "-color_primaries", "bt2020",
            "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc",
            str(out_compare_mov)
        ], verbose=verbose)

    else:
        filt = (
            f"[0:v]setpts=PTS-STARTPTS,"
            f"fps=fps={fps_ffmpeg}:round=near,"
            f"scale=-2:{heat_h}:flags=bicubic"
            f"[v0];"
            f"[1:v]setpts=PTS-STARTPTS,"
            f"fps=fps={fps_ffmpeg}:round=near"
            f"[v1];"
            f"[v0][v1]hstack=inputs=2[v]"
        )

        run([
            ffmpeg, "-hide_banner", "-y",
            "-i", test,
            "-i", heat_mov,
            "-filter_complex", filt,
            "-map", "[v]",
            "-an",
            "-fps_mode", "cfr",
            "-c:v", "prores_ks", "-profile:v", "3", "-q:v", "7",
            "-pix_fmt", "yuv422p10le",
            str(out_compare_mov)
        ], verbose=verbose)


def cleanup_heatmaps_if_success(heat_dir: Path, heat_mov: Path) -> None:
    try:
        if heat_mov.exists() and heat_mov.stat().st_size > 50_000:
            print(f"Heatmap MOV created successfully ({heat_mov.stat().st_size} bytes).")
            print(f"Cleaning up heatmap PNG folder: {heat_dir}")
            shutil.rmtree(heat_dir, ignore_errors=True)
        else:
            print("Heatmap MOV missing or too small — keeping heatmap PNGs for debugging.")
    except Exception as e:
        print(f"Cleanup warning: {e}")


# -------------------------------------------------------------
# CSV legends + options block
# -------------------------------------------------------------

def args_to_kv_lines(args: argparse.Namespace, extra: Dict[str, str]) -> List[str]:
    d = vars(args).copy()
    for k, v in list(d.items()):
        if v is None:
            d[k] = ""
    d.update(extra)

    preferred = [
        "display", "pix_per_deg", "mode", "temp_window", "device",
        "display_res", "chroma_filter",
        "temp_resample", "temp_padding", "dump_channels", "dump_output_dir",
        "no_compare", "keep_work", "limit_frames",
        "verbose",
    ]
    keys: List[str] = []
    for k in preferred:
        if k in d:
            keys.append(k)
    for k in sorted(d.keys()):
        if k not in keys:
            keys.append(k)

    return [f"# {k}={d[k]}" for k in keys]


def write_csv_header_block(w: csv.writer, heatmap_mode: str, options_lines: List[str]) -> None:
    w.writerow(["# CVVDP JOD legend (higher is better; typical practical range ~0–10)"])
    w.writerow(["# ~10.0 : visually indistinguishable / reference quality"])
    w.writerow(["# 9–10  : extremely high quality (differences tiny/rare)"])
    w.writerow(["# 8–9   : small but visible differences"])
    w.writerow(["# 7–8   : mild differences"])
    w.writerow(["# 5–7   : clearly noticeable degradation"])
    w.writerow(["# 3–5   : strong impairment"])
    w.writerow(["# <3    : severe distortion"])
    w.writerow([])
    w.writerow(["# jod_total = native reference vs test"])
    w.writerow(["# jod_total_ref420sim = simulated-420 reference vs test"])
    w.writerow([])

    w.writerow([f"# Heatmap mode: {heatmap_mode}"])
    if heatmap_mode == "supra-threshold":
        w.writerow(["# Heatmap color key (supra-threshold perceptual difference):"])
        w.writerow(["# Dark Blue  : no visible difference"])
        w.writerow(["# Cyan/Green : near detection threshold"])
        w.writerow(["# Yellow     : clearly visible difference"])
        w.writerow(["# Orange     : strong visible difference"])
        w.writerow(["# Red        : highly objectionable difference"])
    elif heatmap_mode == "threshold":
        w.writerow(["# Heatmap color key (threshold detection map):"])
        w.writerow(["# Blue       : below detection threshold"])
        w.writerow(["# Bright     : above detection threshold"])
    elif heatmap_mode == "raw":
        w.writerow(["# Heatmap color key (raw perceptual error energy):"])
        w.writerow(["# Dark       : low error energy"])
        w.writerow(["# Bright     : high error energy"])
    w.writerow(["# Note: heatmap visualizes spatial perceptual error under the selected display model."])
    w.writerow(["# Heatmaps/MOV are generated from the native-reference analysis only."])
    w.writerow([])

    w.writerow(["# Options used (including defaults):"])
    for line in options_lines:
        w.writerow([line])
    w.writerow([])


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="CVVDP per-frame dual-analysis JOD + heatmap PNG seq + MOV.")
    ap.add_argument("ref", nargs="?", help="Reference video file")
    ap.add_argument("test", nargs="?", help="Test video file")
    ap.add_argument("outdir", nargs="?", help="Output directory")

    ap.add_argument("--mode", default="supra-threshold", choices=["supra-threshold", "threshold", "raw"])
    ap.add_argument("--display", default="NBCU_65inch_hdr_pq_2knit")

    ap.add_argument("--pix-per-deg", type=float, default=None,
                    help="Optional override. If omitted, display profile governs pix/deg.")

    ap.add_argument("--temp-window", type=int, default=0,
                    help="Temporal half-window k. Uses frames (i-k):(i+k). 0 = single-frame only.")
    ap.add_argument("--device", default="mps")

    ap.add_argument("--display-res", default="3840x2160",
                    help="Target resolution for TIFF extraction.")
    ap.add_argument("--chroma-filter", default="spline36",
                    choices=["point", "bilinear", "bicubic", "spline16", "spline36", "lanczos"],
                    help="zscale filter used for resize and chroma upsampling during TIFF extraction.")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--limit-frames", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")

    ap.add_argument("--temp-resample", action="store_true",
                    help="Pass through cvvdp --temp-resample (mostly meaningful for video-vs-video).")
    ap.add_argument("--temp-padding", default="replicate", choices=["replicate", "pingpong", "circular"],
                    help="Pass through cvvdp --temp-padding for temporal filters.")
    ap.add_argument("--dump-channels",
                    nargs="+",
                    choices=["temporal", "lpyr", "difference"],
                    default=[],
                    help="Optional cvvdp debug dump stages to emit per frame/window.")
    ap.add_argument("--dump-output-dir",
                    default="",
                    help="Optional directory to preserve cvvdp --dump-channels outputs. "
                         "If omitted, dump outputs are temporary.")

    args = ap.parse_args()

    if not args.ref:
        args.ref = prompt_path("Reference file (drag & drop, then press Enter): ")
    if not args.test:
        args.test = prompt_path("Test file (drag & drop, then press Enter): ")
    if not args.outdir:
        args.outdir = prompt_path("Output directory (drag & drop, then press Enter): ")

    ref = clean_path(args.ref)
    test = clean_path(args.test)
    outdir = Path(clean_path(args.outdir)).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    display_res = parse_display_res(args.display_res)

    if args.dump_output_dir:
        Path(clean_path(args.dump_output_dir)).expanduser().resolve().mkdir(parents=True, exist_ok=True)

    cvvdp_exe = find_cvvdp_exe()
    print(f"\nUsing cvvdp executable: {cvvdp_exe}")

    ref_meta = ffprobe_stream_meta(ref)
    test_meta = ffprobe_stream_meta(test)

    if (is_pq_video(test_meta) or is_pq_video(ref_meta)) and not ffmpeg_has_zscale():
        raise RuntimeError(
            "PQ source/reference detected, but this ffmpeg build does not include zscale.\n"
            "Install/rebuild ffmpeg with libzimg support."
        )

    fps_ffmpeg = ffprobe_r_frame_rate(test)
    fps = r_frame_rate_to_float(fps_ffmpeg)

    print("\nInputs:")
    print(f"  Ref:  {ref}")
    print(f"  Test: {test}")
    print(f"  Out:  {outdir}")
    print("\nffprobe (ref): ", ref_meta)
    print("ffprobe (test):", test_meta)
    print(f"\nUsing FPS: {fps_ffmpeg} (cvvdp fps={fps:.12f})")
    print(f"CVVDP display model: {args.display}")
    if args.pix_per_deg is None:
        print("pix/deg override: (none; display profile governs)")
    else:
        print(f"pix/deg override: {args.pix_per_deg}")
    print(f"temp-window K: {args.temp_window}")
    print(f"display-res: {args.display_res}")
    print(f"chroma-filter: {args.chroma_filter}")
    print(f"temp-resample: {'ON' if args.temp_resample else 'OFF'}")
    print(f"temp-padding: {args.temp_padding}")
    print(f"dump-channels: {args.dump_channels if args.dump_channels else '(none)'}")
    print(f"dump-output-dir: {args.dump_output_dir if args.dump_output_dir else '(temporary only)'}")
    print("ffmpeg:", which_or_raise("ffmpeg"))
    print("ffprobe:", which_or_raise("ffprobe"))
    print("cvvdp :", cvvdp_exe)
    print(f"PQ source detected: {is_pq_video(test_meta)}")
    print(f"PQ reference detected: {is_pq_video(ref_meta)}\n")

    if args.keep_work:
        work_dir = Path(tempfile.mkdtemp(prefix="cvvdp_work_"))
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="cvvdp_work_")
        work_dir = Path(temp_ctx.name)

    try:
        ref_dir_native = work_dir / "ref_native"
        ref_dir_420sim = work_dir / "ref_420sim"
        test_dir = work_dir / "test"

        print("Extracting TIFF sequences (once)…")
        ref_pat_native = extract_tiffs_native(
            ref, ref_dir_native, "refn",
            display_res,
            chroma_filter=args.chroma_filter,
            verbose=args.verbose
        )
        ref_pat_420sim = extract_tiffs_ref420sim(
            ref, ref_dir_420sim, "refs",
            display_res,
            chroma_filter=args.chroma_filter,
            verbose=args.verbose
        )
        test_pat = extract_tiffs_native(
            test, test_dir, "test",
            display_res,
            chroma_filter=args.chroma_filter,
            verbose=args.verbose
        )

        n_ref_native = count_tiffs(ref_dir_native, "refn")
        n_ref_420sim = count_tiffs(ref_dir_420sim, "refs")
        n_test = count_tiffs(test_dir, "test")
        n_frames = min(n_ref_native, n_ref_420sim, n_test)
        if n_frames <= 0:
            raise RuntimeError("No TIFF frames extracted.")

        if args.limit_frames and args.limit_frames > 0:
            n_frames = min(n_frames, args.limit_frames)

        print(f"TIFF frames: ref_native={n_ref_native}, ref_420sim={n_ref_420sim}, test={n_test}, using={n_frames}")
        print(f"Work dir: {work_dir} {'(kept)' if args.keep_work else '(temp)'}\n")

        out_csv = outdir / "metrics_per_frame.csv"
        heat_dir = outdir / f"heatmaps_{args.mode}"
        heat_dir.mkdir(parents=True, exist_ok=True)

        extra_opts = {
            "cvvdp_exe": cvvdp_exe,
            "ref_path": ref,
            "test_path": test,
            "outdir": str(outdir),
            "fps_ffmpeg": fps_ffmpeg,
            "fps_cvvdp_float": f"{fps:.12f}",
            "tiff_scale_to": f"{display_res[0]}x{display_res[1]}",
            "cvvdp_mode": "noninteractive_for_metrics",
            "dual_analysis": "native_and_ref420sim",
        }
        options_lines = args_to_kv_lines(args, extra_opts)

        print(f"Building per-frame CSV: {out_csv}")
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            write_csv_header_block(w, args.mode, options_lines)

            headers = [
                "frame",
                "jod_total",
                "jod_total_ref420sim",
                "time_sec",
                "cvvdp_display_model",
                "pix_per_deg_override",
                "temp_window_k",
                "device",
                "fps_ffmpeg",
                "fps_cvvdp_float",
                "ref_codec", "ref_pix_fmt", "ref_color_transfer", "ref_color_primaries",
                "ref_color_space", "ref_color_range", "ref_width", "ref_height",
                "test_codec", "test_pix_fmt", "test_color_transfer", "test_color_primaries",
                "test_color_space", "test_color_range", "test_width", "test_height",
            ]
            w.writerow(headers)
            f.flush()

            print(f"Generating heatmap PNGs into: {heat_dir}")

            for i in range(n_frames):
                out_png = heat_dir / f"heatmap_{i:06d}.png"

                # Native analysis (also generates heatmap)
                jod_total = run_cvvdp_window_and_heatmap(
                    cvvdp_exe=cvvdp_exe,
                    ref_pat=ref_pat_native,
                    test_pat=test_pat,
                    center_frame=i,
                    temp_window=args.temp_window,
                    display=args.display,
                    device=args.device,
                    fps=fps,
                    heatmap_mode=args.mode,
                    pix_per_deg=args.pix_per_deg,
                    temp_resample=args.temp_resample,
                    temp_padding=args.temp_padding,
                    dump_channels=args.dump_channels,
                    dump_output_dir=args.dump_output_dir,
                    out_png=out_png,
                    verbose=args.verbose
                )

                # Simulated-420 reference analysis (JOD only)
                jod_total_ref420sim = run_cvvdp_jod_only(
                    cvvdp_exe=cvvdp_exe,
                    ref_pat=ref_pat_420sim,
                    test_pat=test_pat,
                    center_frame=i,
                    temp_window=args.temp_window,
                    display=args.display,
                    device=args.device,
                    fps=fps,
                    pix_per_deg=args.pix_per_deg,
                    temp_resample=args.temp_resample,
                    temp_padding=args.temp_padding,
                    verbose=args.verbose
                )

                row = [
                    i,
                    jod_total,
                    jod_total_ref420sim,
                    f"{i / fps:.6f}",
                    args.display,
                    "" if args.pix_per_deg is None else str(args.pix_per_deg),
                    args.temp_window,
                    args.device,
                    fps_ffmpeg,
                    f"{fps:.12f}",
                    ref_meta.get("codec_name", ""),
                    ref_meta.get("pix_fmt", ""),
                    ref_meta.get("color_transfer", ""),
                    ref_meta.get("color_primaries", ""),
                    ref_meta.get("color_space", ""),
                    ref_meta.get("color_range", ""),
                    ref_meta.get("width", ""),
                    ref_meta.get("height", ""),
                    test_meta.get("codec_name", ""),
                    test_meta.get("pix_fmt", ""),
                    test_meta.get("color_transfer", ""),
                    test_meta.get("color_primaries", ""),
                    test_meta.get("color_space", ""),
                    test_meta.get("color_range", ""),
                    test_meta.get("width", ""),
                    test_meta.get("height", ""),
                ]

                w.writerow(row)

                if (i + 1) % 10 == 0 or i == n_frames - 1:
                    f.flush()
                    print(f"  heatmaps+metrics: {i+1}/{n_frames}")

        heat_mov = outdir / f"heatmap_{args.mode}.mov"
        png_pattern = str(heat_dir / "heatmap_%06d.png")
        print(f"\nEncoding heatmap MOV from PNGs (no blending): {heat_mov}")

        encode_png_sequence_to_mov(
            png_pattern,
            fps_ffmpeg,
            heat_mov,
            source_is_pq=is_pq_video(test_meta),
            verbose=args.verbose
        )

        compare_mov = None
        if not args.no_compare:
            compare_mov = outdir / f"compare_test_plus_heatmap_{args.mode}.mov"
            if is_pq_video(test_meta):
                print("Compare MOV mode: source is PQ, auto-upmapping heatmap to PQ/BT.2020 for side-by-side")
            else:
                print("Compare MOV mode: source is not PQ, using standard side-by-side pipeline")
            print(f"Encoding side-by-side MOV: {compare_mov}")
            encode_compare_mov(
                test,
                str(heat_mov),
                compare_mov,
                fps_ffmpeg,
                test_meta=test_meta,
                verbose=args.verbose
            )

        cleanup_heatmaps_if_success(heat_dir, heat_mov)

        print("\nDone.")
        print(f"  CSV:          {out_csv}")
        print(f"  Heatmap MOV:  {heat_mov}")
        if compare_mov:
            print(f"  Compare MOV:  {compare_mov}")
        if args.keep_work:
            print(f"  Kept workdir: {work_dir}")

    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    main()