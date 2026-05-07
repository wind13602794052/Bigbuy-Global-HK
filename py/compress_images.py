#!/usr/bin/env python3
"""
从项目根目录的 image/ 读取图片，压缩并输出到项目根目录的 image_thumb/。
依赖：pip install Pillow easyocr（首次运行会下载 OCR 模型）

输出文件体积不超过 MAX_FILE_BYTES（默认 200KB）。
若 OCR 识别到文字，水印叠放在文字区域中心（压在文字上）；否则水印置于整张图中心。
注意：OCR 只认「画里印出来的字」，不会读硬盘上的文件名（例如 2.19.png 只是路径，图里没写就识别不到）。

水印使用 py/Wechat.jpg，较长边为原图较长边的十分之一，透明度 50%。
调试：OCR_DEBUG=1 时向 stderr 打印每张图识别到的文字片段与锚点。
OCR 使用多路预处理（对比度/锐化/小图放大）+ 宽松检测参数。EasyOCR 须使用
["ch_sim","en"] 或 ["ch_tra","en"]，不可把 ch_sim 与 ch_tra 混在同一 Reader。
无简中结果时会自动再试繁体+英文。仅繁体图可设环境变量 OCR_LANG_ONLY=tra。
仍无结果可调低 OCR_MIN_CONFIDENCE 或设 OCR_DEBUG=1 查看识别片段。

文字识别开关：
  --no-ocr          关闭 OCR，水印始终居中（不加载 EasyOCR，启动更快）
  OCR_ENABLE=0      与 --no-ocr 等价（便于脚本/CI）

用法（在项目根目录）：python py/compress_images.py [--no-ocr]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from PIL import Image, ImageEnhance, ImageOps
except ImportError:
    print("请先安装 Pillow：pip install Pillow", file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "image"
DST_DIR = PROJECT_ROOT / "image_thumb"
WATERMARK_PATH = SCRIPT_DIR / "Wechat.jpg"

MAX_SIDE = 2000
JPEG_QUALITY = 50
PNG_COMPRESSION = 5

MAX_FILE_BYTES = 200 * 1024
MIN_JPEG_QUALITY = 15
SHRINK_FACTOR = 0.88
MAX_SHRINK_ROUNDS = 64
WATERMARK_LONG_EDGE_RATIO = 0.3
WATERMARK_OPACITY = 0.5
OCR_ENABLE = 0
# OCR：识别框置信度下限（略低以提高召回；误检多可调高到 0.2～0.3）
OCR_MIN_CONFIDENCE = 0.12
# 长边小于此像素时，生成放大版专供 OCR（工业图里小字更清晰）
OCR_MIN_LONG_EDGE_FOR_UPSCALE = 880
# 放大后目标长边（上限避免内存暴涨）
OCR_UPSCALE_TARGET_LONG = 1680
OCR_UPSCALE_MAX_FACTOR = 2.8
# EasyOCR 检测：放宽以检出弱对比、小字号（参见 easyocr.readtext 参数）
OCR_DETECT_MIN_SIZE = 5
OCR_DETECT_TEXT_THRESHOLD = 0.42
OCR_DETECT_LOW_TEXT = 0.28
OCR_DETECT_LINK_THRESHOLD = 0.32
OCR_DETECT_MAG_RATIO = 2.4

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="压缩 image 输出到 image_thumb，可选 OCR 将水印对齐到文字区域",
    )
    p.add_argument(
        "--no-ocr",
        action="store_true",
        help="关闭文字识别（不加载 EasyOCR），水印始终在画布中央",
    )
    return p.parse_args()


def env_ocr_enabled() -> bool:
    """环境变量 OCR_ENABLE：默认开启；0/false/no/off 表示关闭。"""
    v = os.environ.get("OCR_ENABLE", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def load_watermark(path: Path) -> Image.Image:
    with Image.open(path) as wm:
        wm.load()
        return wm.convert("RGBA")


def _prepare_watermark_rgba(
    wm: Image.Image, bw: int, bh: int
) -> tuple[Image.Image, int, int]:
    wm_rgba = wm.convert("RGBA")
    ww, wh = wm_rgba.size
    if bw <= 0 or bh <= 0 or ww <= 0 or wh <= 0:
        return wm_rgba, ww, wh

    base_long = max(bw, bh)
    wm_long = max(ww, wh)
    target_long = max(1, round(base_long * WATERMARK_LONG_EDGE_RATIO))
    scale = target_long / wm_long
    nw = max(1, round(ww * scale))
    nh = max(1, round(wh * scale))
    wm_rgba = wm_rgba.resize((nw, nh), Image.Resampling.LANCZOS)
    ww, wh = wm_rgba.size

    if ww > bw or wh > bh:
        fit = min(bw / ww, bh / wh)
        nw2 = max(1, int(ww * fit))
        nh2 = max(1, int(wh * fit))
        wm_rgba = wm_rgba.resize((nw2, nh2), Image.Resampling.LANCZOS)
        ww, wh = wm_rgba.size

    r, g, b, alpha = wm_rgba.split()
    alpha = alpha.point(lambda p: min(255, int(round(p * WATERMARK_OPACITY))))
    wm_rgba = Image.merge("RGBA", (r, g, b, alpha))
    return wm_rgba, ww, wh


def apply_watermark(
    base: Image.Image,
    wm: Image.Image,
    *,
    anchor_center: tuple[int, int] | None = None,
) -> Image.Image:
    """
    叠加水印。anchor_center 为 (cx, cy) 时，水印几何中心对齐该点（用于压在文字上）；
    为 None 时置于画布正中央。
    """
    base_rgba = base.convert("RGBA")
    bw, bh = base_rgba.size
    wm_rgba, ww, wh = _prepare_watermark_rgba(wm, bw, bh)

    if anchor_center is None:
        cx, cy = bw // 2, bh // 2
    else:
        cx, cy = anchor_center

    x = int(round(cx - ww / 2))
    y = int(round(cy - wh / 2))
    x = max(0, min(x, bw - ww))
    y = max(0, min(y, bh - wh))

    out = base_rgba.copy()
    out.paste(wm_rgba, (x, y), wm_rgba)
    return out


def _map_box_to_im_coords(
    box: list, sx: float, sy: float
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for px, py in box:
        xs.append(float(px) * sx)
        ys.append(float(py) * sy)
    return xs, ys


def _iter_ocr_numpy_variants(im: Image.Image):
    """生成多路输入：(rgb ndarray, sx, sy)，检测坐标乘以 sx/sy 映射回原图 im。"""
    import numpy as np

    rgb = im.convert("RGB")
    w, h = rgb.size
    le = max(w, h)

    yield np.asarray(rgb), 1.0, 1.0

    try:
        ac = ImageOps.autocontrast(rgb, cutoff=2)
        yield np.asarray(ac), 1.0, 1.0
    except Exception:
        pass

    enh = ImageEnhance.Contrast(rgb).enhance(1.12)
    yield np.asarray(enh.convert("RGB")), 1.0, 1.0

    if le < OCR_MIN_LONG_EDGE_FOR_UPSCALE:
        factor = min(OCR_UPSCALE_MAX_FACTOR, OCR_UPSCALE_TARGET_LONG / le)
        nw = max(1, int(round(w * factor)))
        nh = max(1, int(round(h * factor)))
        big = rgb.resize((nw, nh), Image.Resampling.LANCZOS)
        sx, sy = w / nw, h / nh
        yield np.asarray(big), sx, sy
        try:
            yield np.asarray(ImageOps.autocontrast(big, cutoff=2)), sx, sy
        except Exception:
            pass
        yield np.asarray(ImageEnhance.Sharpness(big).enhance(1.2).convert("RGB")), sx, sy


def _readtext_detect(reader, arr):
    """调用 EasyOCR，优先使用利于小字/弱对比的检测参数。"""
    kw = {
        "detail": 1,
        "paragraph": False,
        "min_size": OCR_DETECT_MIN_SIZE,
        "text_threshold": OCR_DETECT_TEXT_THRESHOLD,
        "low_text": OCR_DETECT_LOW_TEXT,
        "link_threshold": OCR_DETECT_LINK_THRESHOLD,
        "mag_ratio": OCR_DETECT_MAG_RATIO,
        "canvas_size": 2560,
    }
    try:
        return reader.readtext(arr, **kw)
    except TypeError:
        return reader.readtext(arr, detail=1, paragraph=False)


def _readtext_detect_loose(reader, arr):
    """更宽松的一轮，用于前序无结果时补救。"""
    kw = {
        "detail": 1,
        "paragraph": False,
        "min_size": 3,
        "text_threshold": 0.32,
        "low_text": 0.2,
        "link_threshold": 0.25,
        "mag_ratio": 2.9,
        "canvas_size": 2560,
    }
    try:
        return reader.readtext(arr, **kw)
    except TypeError:
        return reader.readtext(arr, detail=1, paragraph=False)


def build_ocr_anchor_fn() -> tuple:
    """
    返回 (ocr_run, 说明文案)。ocr_run 签名为：

        (im) -> (anchor, texts)
    不可用则为 (None, 原因)。

    EasyOCR 限制：ch_sim 与 ch_tra 不能放在同一个 Reader 里。
    策略：先加载 ["ch_sim","en"]；若无任何文字框，再懒加载 ["ch_tra","en"] 扫一遍。
    环境变量 OCR_LANG_ONLY=tra 时仅使用 ["ch_tra","en"]（纯繁体场景）。
    """
    try:
        import easyocr
    except ImportError:
        return None, "未安装 easyocr，水印将居中（pip install easyocr）"

    only_tra = os.environ.get("OCR_LANG_ONLY", "").strip().lower() in (
        "tra",
        "1",
        "true",
        "yes",
    )

    try:
        if only_tra:
            reader_primary = easyocr.Reader(
                ["ch_tra", "en"],
                gpu=False,
                verbose=False,
            )
        else:
            reader_primary = easyocr.Reader(
                ["ch_sim", "en"],
                gpu=False,
                verbose=False,
            )
    except Exception as e:
        return None, f"EasyOCR 初始化失败，水印将居中：{e}"

    reader_tra: object | None = None
    reader_tra_init_failed = False

    def get_reader_tra() -> object | None:
        nonlocal reader_tra, reader_tra_init_failed
        if only_tra:
            return None
        if reader_tra_init_failed:
            return None
        if reader_tra is not None:
            return reader_tra
        try:
            reader_tra = easyocr.Reader(
                ["ch_tra", "en"],
                gpu=False,
                verbose=False,
            )
        except Exception:
            reader_tra_init_failed = True
            return None
        return reader_tra

    def ocr_run(
        im: Image.Image,
    ) -> tuple[tuple[int, int] | None, list[str]]:
        all_xs: list[float] = []
        all_ys: list[float] = []
        texts: list[str] = []
        seen_text: set[str] = set()

        def consume_results(results, sx: float, sy: float) -> None:
            for item in results:
                if len(item) < 3:
                    continue
                box, text, conf = item[0], item[1], float(item[2])
                if conf < OCR_MIN_CONFIDENCE:
                    continue
                if not box or len(box) < 4:
                    continue
                xs, ys = _map_box_to_im_coords(box, sx, sy)
                all_xs.extend(xs)
                all_ys.extend(ys)
                if isinstance(text, str):
                    t = text.strip()
                    if t and t not in seen_text:
                        seen_text.add(t)
                        texts.append(t)

        def scan_with_reader(reader: object, variants: list) -> None:
            for arr, sx, sy in variants:
                try:
                    consume_results(_readtext_detect(reader, arr), sx, sy)
                except Exception:
                    continue
            if not all_xs:
                for arr, sx, sy in variants:
                    try:
                        consume_results(
                            _readtext_detect_loose(reader, arr), sx, sy
                        )
                    except Exception:
                        continue

        variants = list(_iter_ocr_numpy_variants(im))
        scan_with_reader(reader_primary, variants)

        if not all_xs and not only_tra:
            tra = get_reader_tra()
            if tra is not None:
                scan_with_reader(tra, variants)

        if not all_xs:
            return None, texts
        l, r = min(all_xs), max(all_xs)
        t, b = min(all_ys), max(all_ys)
        cx = int(round((l + r) / 2))
        cy = int(round((t + b) / 2))
        return (cx, cy), texts

    label = "已启用 EasyOCR（ch_sim+en；无结果时尝试 ch_tra+en）"
    if only_tra:
        label = "已启用 EasyOCR（仅 ch_tra+en，OCR_LANG_ONLY=tra）"
    return ocr_run, label


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
    args = parse_args()
    use_ocr = env_ocr_enabled() and not args.no_ocr

    if not SRC_DIR.is_dir():
        print(
            f"未找到源目录: {SRC_DIR}\n请先创建 image 文件夹并放入图片。",
            file=sys.stderr,
        )
        return 1

    if not WATERMARK_PATH.is_file():
        print(
            f"未找到水印文件: {WATERMARK_PATH}\n请将 Wechat.jpg 放在 py 目录下（与 compress_images.py 同级）。",
            file=sys.stderr,
        )
        return 1

    try:
        watermark = load_watermark(WATERMARK_PATH)
    except OSError as e:
        print(f"无法读取水印图片: {WATERMARK_PATH} ({e})", file=sys.stderr)
        return 1

    if use_ocr:
        ocr_run, ocr_status = build_ocr_anchor_fn()
        print(ocr_status, flush=True)
    else:
        ocr_run = None
        reason = []
        if args.no_ocr:
            reason.append("--no-ocr")
        if not env_ocr_enabled():
            reason.append("OCR_ENABLE=0")
        detail = "、".join(reason) if reason else "配置"
        print(f"OCR 已关闭（{detail}），水印居中", flush=True)

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

                anchor: tuple[int, int] | None = None
                wm_mode = "居中"
                ocr_texts: list[str] = []
                if ocr_run is not None:
                    anchor, ocr_texts = ocr_run(thumb)
                    if anchor is not None:
                        wm_mode = "文字上"
                    tag = f"{tag}+OCR({wm_mode})"
                    if os.environ.get("OCR_DEBUG", "").strip() in ("1", "true", "yes"):
                        preview = ocr_texts[:8]
                        print(
                            f"  [OCR_DEBUG] {path.name} 识别块={len(ocr_texts)} "
                            f"锚点={anchor} 片段={preview!r}",
                            file=sys.stderr,
                            flush=True,
                        )
                thumb = apply_watermark(thumb, watermark, anchor_center=anchor)
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
