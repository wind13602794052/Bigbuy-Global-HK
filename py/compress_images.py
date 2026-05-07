#!/usr/bin/env python3
"""
将 image/ 下的图片压缩并输出到 image_thumb/
依赖：pip install Pillow

输出文件体积不超过 MAX_FILE_BYTES（默认 200KB）。
源文件本身已小于等于该上限时，不进行缩放压缩，但仍会叠加水印并保存。
水印使用脚本同目录下的 Wechat.jpg，居中叠放；水印较长边为原图较长边的十分之一。

用法：python compress_images.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("请先安装 Pillow：pip install Pillow", file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "image"
DST_DIR = SCRIPT_DIR / "image_thumb"
WATERMARK_PATH = SCRIPT_DIR / "Wechat.jpg"

MAX_SIDE = 1200
JPEG_QUALITY = 50
PNG_COMPRESSION = 5

# 压缩后单文件最大字节数（200 KB）
MAX_FILE_BYTES = 200 * 1024
# JPEG/WebP 质量下限（再降观感较差）
MIN_JPEG_QUALITY = 15
# 仍超限时缩小尺寸的比例（每轮）
SHRINK_FACTOR = 0.88
# 防止异常图片死循环
MAX_SHRINK_ROUNDS = 64
# 水印较长边 = 原图较长边 × 该比例（十分之一）
WATERMARK_LONG_EDGE_RATIO = 0.1

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def load_watermark(path: Path) -> Image.Image:
    with Image.open(path) as wm:
        wm.load()
        return wm.convert("RGBA")


def apply_center_watermark(base: Image.Image, wm: Image.Image) -> Image.Image:
    """将水印置于 base 中央；水印较长边缩放为原图较长边的 WATERMARK_LONG_EDGE_RATIO。"""
    base_rgba = base.convert("RGBA")
    wm_rgba = wm.convert("RGBA")
    bw, bh = base_rgba.size
    ww, wh = wm_rgba.size
    if bw <= 0 or bh <= 0:
        return base_rgba
    if ww <= 0 or wh <= 0:
        return base_rgba

    base_long = max(bw, bh)
    wm_long = max(ww, wh)
    target_long = max(1, round(base_long * WATERMARK_LONG_EDGE_RATIO))
    scale = target_long / wm_long
    nw = max(1, round(ww * scale))
    nh = max(1, round(wh * scale))
    wm_rgba = wm_rgba.resize((nw, nh), Image.Resampling.LANCZOS)
    ww, wh = wm_rgba.size

    # 极端长宽比下仍保证不超出画布
    if ww > bw or wh > bh:
        fit = min(bw / ww, bh / wh)
        nw2 = max(1, int(ww * fit))
        nh2 = max(1, int(wh * fit))
        wm_rgba = wm_rgba.resize((nw2, nh2), Image.Resampling.LANCZOS)
        ww, wh = wm_rgba.size

    x = (bw - ww) // 2
    y = (bh - wh) // 2
    out = base_rgba.copy()
    out.paste(wm_rgba, (x, y), wm_rgba)
    return out


def scale_down(im: Image.Image, max_side: int) -> Image.Image:
    w, h = im.size
    long_edge = max(w, h)
    if long_edge <= max_side:
        return im.copy()

    scale = max_side / long_edge
    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def shrink_im(im: Image.Image, factor: float) -> Image.Image:
    w, h = im.size
    nw = max(1, round(w * factor))
    nh = max(1, round(h * factor))
    if nw >= w and nh >= h and factor < 1.0:
        nw = max(1, w - 1)
        nh = max(1, h - 1)
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def _to_rgb_for_jpeg(im: Image.Image) -> Image.Image:
    if im.mode in ("RGB", "L"):
        return im.convert("RGB") if im.mode == "L" else im
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    if im.mode == "P" and "transparency" in im.info:
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


def _png_for_save(im: Image.Image) -> Image.Image:
    if im.mode not in ("RGBA", "RGB", "L", "P"):
        return im.convert("RGBA")
    return im


def save_under_max_bytes(im: Image.Image, path: Path, ext: str) -> bool:
    """保存图片，确保 path 对应文件大小不超过 MAX_FILE_BYTES。"""
    ext_lower = ext.lower()
    cur = im.copy()

    for _ in range(MAX_SHRINK_ROUNDS):
        if ext_lower in (".jpg", ".jpeg"):
            for q in range(JPEG_QUALITY, MIN_JPEG_QUALITY - 1, -5):
                try:
                    out = _to_rgb_for_jpeg(cur)
                    out.save(path, format="JPEG", quality=q, optimize=True)
                    if path.stat().st_size <= MAX_FILE_BYTES:
                        return True
                except OSError:
                    return False

        elif ext_lower == ".webp":
            for q in range(JPEG_QUALITY, MIN_JPEG_QUALITY - 1, -5):
                try:
                    cur.save(path, format="WEBP", quality=q, method=6)
                    if path.stat().st_size <= MAX_FILE_BYTES:
                        return True
                except OSError:
                    return False

        elif ext_lower == ".png":
            png = _png_for_save(cur)
            for level in range(PNG_COMPRESSION, 10):
                try:
                    png.save(
                        path,
                        format="PNG",
                        compress_level=level,
                        optimize=True,
                    )
                    if path.stat().st_size <= MAX_FILE_BYTES:
                        return True
                except OSError:
                    return False

        elif ext_lower == ".gif":
            try:
                cur.save(path, format="GIF", optimize=True)
                if path.stat().st_size <= MAX_FILE_BYTES:
                    return True
            except OSError:
                return False

        else:
            return False

        cur = shrink_im(cur, SHRINK_FACTOR)

    # 极端情况：继续缩小并用最低压缩参数保存，直到不超过上限或像素无法再缩
    while True:
        try:
            if ext_lower in (".jpg", ".jpeg"):
                _to_rgb_for_jpeg(cur).save(
                    path, format="JPEG", quality=MIN_JPEG_QUALITY, optimize=True
                )
            elif ext_lower == ".webp":
                cur.save(
                    path, format="WEBP", quality=MIN_JPEG_QUALITY, method=6
                )
            elif ext_lower == ".png":
                _png_for_save(cur).save(
                    path, format="PNG", compress_level=9, optimize=True
                )
            elif ext_lower == ".gif":
                cur.save(path, format="GIF", optimize=True)
            else:
                return False
        except OSError:
            return False

        if path.stat().st_size <= MAX_FILE_BYTES:
            return True
        if max(cur.size) <= 1:
            return path.is_file()
        cur = shrink_im(cur, SHRINK_FACTOR)


def main() -> int:
    if not SRC_DIR.is_dir():
        print(
            f"未找到源目录: {SRC_DIR}\n请先创建 image 文件夹并放入图片。",
            file=sys.stderr,
        )
        return 1

    if not WATERMARK_PATH.is_file():
        print(
            f"未找到水印文件: {WATERMARK_PATH}\n请将 Wechat.jpg 放在项目根目录（与脚本同级）。",
            file=sys.stderr,
        )
        return 1

    try:
        watermark = load_watermark(WATERMARK_PATH)
    except OSError as e:
        print(f"无法读取水印图片: {WATERMARK_PATH} ({e})", file=sys.stderr)
        return 1

    DST_DIR.mkdir(parents=True, exist_ok=True)

    ok = 0
    fail = 0

    for path in sorted(SRC_DIR.iterdir()):
        if not path.is_file():
            continue
        ext = path.suffix
        if ext.lower() not in ALLOWED_EXT:
            continue

        out_path = DST_DIR / f"{path.stem}{ext}"

        try:
            src_bytes = path.stat().st_size
        except OSError as e:
            print(f"跳过（无法读取文件信息）: {path} ({e})", file=sys.stderr)
            fail += 1
            continue

        try:
            with Image.open(path) as im:
                im.load()
                if getattr(im, "is_animated", False):
                    im.seek(0)
                if src_bytes <= MAX_FILE_BYTES:
                    thumb = im.copy()
                    tag = "仅水印（原≤200KB，未缩放）"
                else:
                    thumb = scale_down(im, MAX_SIDE)
                    tag = "压缩+水印"
                thumb = apply_center_watermark(thumb, watermark)
        except OSError:
            print(f"跳过（无法读取）: {path}", file=sys.stderr)
            fail += 1
            continue

        if save_under_max_bytes(thumb, out_path, ext):
            size_kb = out_path.stat().st_size / 1024
            print(f"OK [{tag}]: {path.name} ({size_kb:.1f} KB)")
            ok += 1
        else:
            print(f"保存失败: {out_path}", file=sys.stderr)
            fail += 1

    print(f"完成：成功 {ok}，失败 {fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
