from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import os


def save_checkpoint(ckpt_dir, model):
    saved_state = {
      'model': model.state_dict(),
    }
    torch.save(saved_state, ckpt_dir)


# --------------------------------------------------------------------------- #
#                                  字体加载                                    #
# --------------------------------------------------------------------------- #
def _load_font(size: int = 18):
    """优先加载等宽 TTF, 找不到则回退到 PIL 默认字体。"""
    candidates = [
        "C:/Windows/Fonts/consola.ttf",                                    # Windows Consolas
        "C:/Windows/Fonts/cour.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",             # Linux 常见
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default()                                    # 兜底
    except Exception:
        return None


def _text_width(font, s: str) -> float:
    """兼容不同 Pillow 版本的文本像素宽度获取。"""
    if font is None:
        return len(s) * (8 * 0.6)
    try:
        return font.getlength(s)
    except AttributeError:
        try:
            return font.getbbox(s)[2]
        except Exception:
            return len(s) * 8.0


def _font_metrics(font, size: int):
    """返回 (char_h, space_w)。"""
    if font is None:
        return size + 4, size * 0.4
    try:
        asc, desc = font.getmetrics()
        char_h = asc + desc + 2
    except Exception:
        char_h = size + 4
    space_w = _text_width(font, " ")
    if space_w <= 0:
        space_w = size * 0.4
    return char_h, space_w


# --------------------------------------------------------------------------- #
#                      把任意结构归一化为 "一帧纯文本"                         #
# --------------------------------------------------------------------------- #
def _to_text(frame, tokenizer=None) -> str:
    """
    frame 可能是:
      - str                                  -> 直接返回
      - torch.Tensor [n, L] 或 [L]           -> 取第 0 个样本解码 (skip_special_tokens=False, 保留 [MASK])
      - list[str]  (batch_decode 的返回)      -> 取第 0 个
      - list[Tensor] / list[list]            -> 递归取第 0 个
    """
    if isinstance(frame, str):
        return frame
    # 惰性导入 torch, 避免纯粹只是解码文本时也强依赖 torch
    try:
        import torch as _torch
    except Exception:
        _torch = None
    if _torch is not None and isinstance(frame, _torch.Tensor):
        t = frame
        if t.ndim > 1:
            t = t[0]
        if tokenizer is not None:
            try:
                return tokenizer.decode(t.tolist(), skip_special_tokens=False)
            except Exception:
                return tokenizer.decode(t.tolist())
        return str(t.tolist())
    if isinstance(frame, (list, tuple)):
        if len(frame) == 0:
            return ""
        return _to_text(frame[0], tokenizer)
    return str(frame)


def _normalize_samples(samples, tokenizer=None):
    """把 ar / default 的 intermediate samples 统一拍平成 list[str]。"""
    if samples is None:
        return []
    out = []
    for fr in samples:
        out.append(_to_text(fr, tokenizer))
    return out


def _split_tokens(text: str):
    """按空白切分; [MASK] 无内部空格, 会作为整体保留。"""
    return text.split()


# --------------------------------------------------------------------------- #
#                            贪心逐词折行 + 记录坐标                            #
# --------------------------------------------------------------------------- #
def _layout(tokens, font, space_w, max_w):
    """
    返回 lines: list[ list[ (token, x_offset) ] ]
    每个 token 在所属行内的 x 像素偏移已知, 便于精确绘制 [MASK] 高亮背景。
    """
    lines = []
    cur = []
    x = 0.0
    for tok in tokens:
        w = _text_width(font, tok)
        if cur and (x + space_w + w) > max_w:          # 当前行放不下, 换行
            lines.append(cur)
            cur = [(tok, 0.0)]
            x = w
        else:
            if cur:
                x += space_w
            cur.append((tok, x))
            x += w
    if cur:
        lines.append(cur)
    return lines


# --------------------------------------------------------------------------- #
#                                   绘制单帧                                   #
# --------------------------------------------------------------------------- #
def _render_frame(lines, canvas_w, canvas_h, char_h, space_w, font,
                  header_text, pad=14,
                  bg=(252, 252, 252),
                  mask_bg=(255, 170, 170), mask_fg=(180, 0, 0),
                  text_color=(45, 45, 45),
                  header_color=(0, 0, 0),
                  header_h=34):
    img = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(img)

    # 顶部标题
    if header_text:
        draw.text((pad, 8), header_text, fill=header_color, font=font)

    y0 = header_h
    for line in lines:
        # 先画 [MASK] 高亮背景, 再画文字 (保证文字盖在背景之上)
        for tok, xoff in line:
            if "[MASK]" in tok:
                w = _text_width(font, tok)
                draw.rectangle([pad + xoff, y0, pad + xoff + w, y0 + char_h],
                                fill=mask_bg)
        for tok, xoff in line:
            draw.text((pad + xoff, y0), tok, fill=text_color, font=font)
        y0 += char_h
    return img


# --------------------------------------------------------------------------- #
#                          均匀抽帧 (避免 GIF 过长)                            #
# --------------------------------------------------------------------------- #
def _subsample(items, max_frames):
    n = len(items)
    if (not max_frames) or n <= max_frames:
        return items
    idx = np.linspace(0, n - 1, max_frames).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [items[i] for i in idx]


# --------------------------------------------------------------------------- #
#                                  对外入口                                    #
# --------------------------------------------------------------------------- #
def save_intermediate_gif(
    samples,
    gif_path,
    tokenizer=None,
    title="samples",
    width_px: int = 1000,
    font_size: int = 18,
    duration: int = 450,
    max_frames: int = 64,
    loop: int = 0,
):
    """
    把 intermediate samples 保存为动态 GIF。

    参数
    ----
    samples      : ar_intermediate_samples 或 default_intermediate_samples
    gif_path     : 输出 gif 路径
    tokenizer    : 仅当某帧仍是 Tensor 时才会用到 (兜底解码), 一般可传 None
    title        : 每帧左上角标题
    width_px     : 文本区像素宽度 (折行用)
    font_size    : 字号
    duration     : 每帧停留毫秒
    max_frames   : 帧数上限, 超过会均匀抽帧 (default_sample 有 130 帧, 抽到 64 帧以内)
    loop         : GIF 循环, 0=无限循环
    """
    texts = _normalize_samples(samples, tokenizer)
    if not texts:
        print(f"[save_intermediate_gif] samples 为空, 跳过: {gif_path}")
        return

    texts = _subsample(texts, max_frames)
    n_frames = len(texts)

    font = _load_font(font_size)
    char_h, space_w = _font_metrics(font, font_size)

    pad = 14
    header_h = 34
    text_w = width_px - 2 * pad

    # 先把所有帧折好行, 确定画布高度 (取最大行数, 所有帧等高, GIF 才稳定)
    all_layouts = []
    max_lines = 0
    for t in texts:
        toks = _split_tokens(t)
        lines = _layout(toks, font, space_w, text_w)
        all_layouts.append(lines)
        max_lines = max(max_lines, len(lines))

    canvas_w = width_px
    canvas_h = header_h + max_lines * char_h + pad

    frames = []
    for i, lines in enumerate(all_layouts):
        header = f"{title}  |  frame {i+1}/{n_frames}"
        img = _render_frame(lines, canvas_w, canvas_h, char_h, space_w, font,
                            header_text=header, pad=pad, header_h=header_h)
        frames.append(img)

    os.makedirs(os.path.dirname(os.path.abspath(gif_path)), exist_ok=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        disposal=2,
    )
    print(f"[save_intermediate_gif] 已保存: {gif_path}  ({n_frames} frames)")
