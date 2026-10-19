#!/usr/bin/env python3
"""Act3D project intro video.

Builds 1920x1080 frames with PIL and pipes raw RGB to ffmpeg. Episode frames are
the real rendered views the policy requested during evaluation
(/raid/icy/tmp/movedist/eval256_v40b_s100_scanqa_views). No ablation content.

    python3 make_video.py            -> static/video/act3d.mp4
"""
import glob, json, math, os, re, subprocess, sys
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static/video/act3d.mp4")
VIEWS = "/raid/icy/tmp/movedist/eval256_v40b_s100_scanqa_views/views"
JSONL = "/raid/icy/tmp/movedist/eval256_v40b_s100_scanqa.jsonl"
FDIR = "/usr/share/fonts/truetype/dejavu/"
ROUND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static/fonts/Nunito.ttf")

INK, SOFT, LINE = (29, 37, 48), (110, 122, 138), (226, 232, 240)
BG, CARD = (255, 255, 255), (246, 248, 251)
BLUE, TEAL, GREEN, PURPLE = (124, 151, 238), (38, 150, 170), (63, 157, 106), (155, 107, 191)
ORANGE = TEAL            # no warm tones in the video; kept as an alias for old call sites
GLOW = (120, 205, 255)
ACCENT = (46, 95, 163)

_fcache = {}
def font(sz, bold=False, mono=False):
    """Nunito (rounded, Apple-ish) for everything but the monospace bits."""
    key = (sz, bold, mono)
    if key in _fcache: return _fcache[key]
    if mono:
        f = ImageFont.truetype(FDIR + "DejaVuSansMono.ttf", sz)
    else:
        f = ImageFont.truetype(ROUND, sz)
        try: f.set_variation_by_axes([800 if bold else 500])
        except Exception: pass
    _fcache[key] = f
    return f

def chain_w(d, words, f, gap=18):
    w = sum(d.textlength(x, font=f) for x in words)
    h = f.size * 0.30
    return w + (len(words) - 1) * (2 * gap + h * 2.1 + 7)

def chain(d, xy, words, f, col=INK, arrow_col=None, gap=18):
    """words joined by small drawn arrows (Nunito has no U+2192 glyph)"""
    x, y = xy
    arrow_col = arrow_col or col
    for i, w_ in enumerate(words):
        d.text((x, y), w_, font=f, fill=col)
        x += d.textlength(w_, font=f)
        if i < len(words) - 1:
            x += gap
            h = f.size * 0.30
            ay = y + f.size * 0.68       # ink middle of the words, not the em box
            d.line([(x, ay), (x + h * 2.1, ay)], fill=arrow_col, width=4)
            d.polygon([(x + h * 2.1 + 7, ay),
                       (x + h * 1.5, ay - h * 0.62),
                       (x + h * 1.5, ay + h * 0.62)], fill=arrow_col)
            x += h * 2.1 + 7 + gap
    return x

def mfont(sz, bold=False):
    """DejaVu for the formula: Nunito has no Greek glyphs"""
    return ImageFont.truetype(FDIR + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), sz)

def draw_runs(d, x, y, runs, f, fsub, col=INK):
    """text with subscripts; returns the x it ended at"""
    for txt, sub in runs:
        d.text((x, y), txt, font=f, fill=col)
        x += d.textlength(txt, font=f)
        if sub:
            d.text((x + 1, y + f.size * 0.42), sub, font=fsub, fill=col)
            x += d.textlength(sub, font=fsub) + 4
    return x

def brace(d, x0, x1, y, col, label, f):
    """flat under-brace with end ticks and a centred label"""
    mid = (x0 + x1) / 2
    d.line([(x0, y), (x1, y)], fill=col, width=4)
    for xx in (x0, x1):
        d.line([(xx, y), (xx, y - 12)], fill=col, width=4)
    d.line([(mid, y), (mid, y + 14)], fill=col, width=4)
    d.text((mid, y + 46), label, font=f, fill=col, anchor="mm")

def robot(img, x, y, sz=1.0, col=(46, 95, 163), skin=(235, 240, 252)):
    """small friendly robot avatar, drawn at (x, y) = top-left"""
    d = ImageDraw.Draw(img)
    w, h = int(104 * sz), int(96 * sz)
    d.line([(x + w // 2, y - int(20 * sz)), (x + w // 2, y + int(6 * sz))], fill=col, width=int(5 * sz))
    r = int(9 * sz)
    d.ellipse([x + w // 2 - r, y - int(20 * sz) - r, x + w // 2 + r, y - int(20 * sz) + r], fill=col)
    d.rounded_rectangle([x, y, x + w, y + h], int(26 * sz), fill=skin, outline=col, width=int(5 * sz))
    ew, eh = int(17 * sz), int(21 * sz)
    for ex in (x + int(26 * sz), x + w - int(26 * sz) - ew):
        d.rounded_rectangle([ex, y + int(28 * sz), ex + ew, y + int(28 * sz) + eh], int(7 * sz), fill=col)
    d.arc([x + int(30 * sz), y + int(52 * sz), x + w - int(30 * sz), y + int(80 * sz)],
          start=15, end=165, fill=col, width=int(5 * sz))
    for sx in (x - int(13 * sz), x + w + int(1 * sz)):
        d.rounded_rectangle([sx, y + int(34 * sz), sx + int(12 * sz), y + int(62 * sz)],
                            int(6 * sz), fill=skin, outline=col, width=int(4 * sz))
    return (x + w, y + h // 2)

UNIT = math.pi / 6                 # one rotation unit, as the walker performs it

def _parse_act(a):
    m = re.match(r"([a-z_]+)\+?(\d*)", (a or "").strip())
    if not m: return None, 0
    return m.group(1), max(1, int(m.group(2) or 1))

def walk_pose(steps, k, u):
    """where the walker is after steps[:k] and a fraction u of steps[k]

    The same action vocabulary the policy emits: turn_left/turn_right, forward/
    backward, left/right strafes, up/down. Yaw is drawn by squashing the figure,
    so a turn away from the viewer reads as the body going edge-on and then
    showing its back.
    """
    x = y = 0.0; sc = 1.0; th = 0.0
    for j in range(k + 1):
        if j >= len(steps): break
        kind, n = _parse_act(steps[j]["act"])
        f = 1.0 if j < k else u
        if   kind == "turn_right": th -= n * UNIT * f
        elif kind == "turn_left":  th += n * UNIT * f
        elif kind == "right":      x += n * 10 * f
        elif kind == "left":       x -= n * 10 * f
        elif kind == "forward":    sc *= 1 + 0.05 * n * f
        elif kind == "backward":   sc /= 1 + 0.05 * n * f
        elif kind == "up":         y -= n * 6 * f
        elif kind == "down":       y += n * 6 * f
    kind = _parse_act(steps[k]["act"])[0] if k < len(steps) else None
    return x, y, sc, th, kind in ("left", "right", "forward", "backward")

def walker(d, x0, by, h, steps, k, u, col=ACCENT):
    """a small cartoon person acting out the move the policy actually made

    Everything is stroked white underneath so it reads over a rendered view.
    """
    xo, yo, scl, th, moving = walk_pose(steps, k, u)
    h = h * scl
    x, y = x0 + xo, by + yo
    sw = math.sin(u * math.pi * 4) if moving else 0.0
    y -= abs(sw) * h * 0.03
    sx = max(0.30, abs(math.cos(th)))      # never fully edge-on: the head must stay readable
    hd, sho, hip = y - h * 0.84, y - h * 0.63, y - h * 0.34
    rh, hw = h * 0.165, h * 0.13 * sx

    def limb(p0, p1, w):
        d.line([p0, p1], fill=(255, 255, 255), width=int(w) + 6)
        d.line([p0, p1], fill=col, width=int(w))
        for p in (p0, p1):
            r = w / 2
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=col)

    swing = sw * h * 0.22
    limb((x - hw * 0.8, sho + h * 0.02), (x - hw - swing * 0.7, sho + h * 0.22), h * 0.085)
    limb((x - hw * 0.4, hip), (x - swing * 0.9, y), h * 0.1)
    d.rounded_rectangle([x - hw - 3, sho - h * 0.06 - 3, x + hw + 3, hip + 3],
                        radius=(hw + 3), fill=(255, 255, 255))
    d.rounded_rectangle([x - hw, sho - h * 0.06, x + hw, hip], radius=hw, fill=col)
    limb((x + hw * 0.4, hip), (x + swing * 0.9, y), h * 0.1)
    limb((x + hw * 0.8, sho + h * 0.02), (x + hw + swing * 0.7, sho + h * 0.22), h * 0.085)

    d.ellipse([x - rh * sx - 4, hd - rh - 4, x + rh * sx + 4, hd + rh + 4], fill=(255, 255, 255))
    d.ellipse([x - rh * sx, hd - rh, x + rh * sx, hd + rh], fill=col)
    if math.cos(th) > 0.34:                # facing us enough to show an eye
        ex = x + math.sin(th) * rh * 0.5
        d.ellipse([ex - rh * 0.2, hd - rh * 0.3, ex + rh * 0.2, hd + rh * 0.1],
                  fill=(255, 255, 255))


def rrect(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)

def arrow(d, p0, p1, col, w=10, bow=0.22, head=34):
    """fat hand-drawn-ish curved arrow with a chunky head"""
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    L = max(1e-6, math.hypot(dx, dy))
    cx, cy = mx - dy * bow, my + dx * bow            # quadratic control point
    pts, N = [], 26
    for i in range(N + 1):
        t = i / N
        bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
        by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
        pts.append((bx, by))
    for i in range(N):                                # taper: thin start, fat end
        ww = w * (0.45 + 0.55 * i / N)
        d.line([pts[i], pts[i + 1]], fill=col, width=int(ww), joint="curve")
    ax, ay = pts[-1]; bx, by = pts[-3]
    ang = math.atan2(ay - by, ax - bx)
    for sgn in (1, -1):
        a2 = ang + sgn * 2.5
        d.line([(ax, ay), (ax + head * math.cos(a2), ay + head * math.sin(a2))],
               fill=col, width=int(w * 1.1), joint="curve")
    d.ellipse([ax - w * 0.55, ay - w * 0.55, ax + w * 0.55, ay + w * 0.55], fill=col)

def bubble(img, box, text_lines, f, fill=(238, 243, 255), outline=(124, 151, 238), tail="bl"):
    """rounded thought bubble with two trailing dots"""
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    rrect(d, box, 26, fill=fill, outline=outline, width=3)
    for i, ln in enumerate(text_lines):
        d.text((x0 + 26, y0 + 22 + i * (f.size + 9)), ln, font=f, fill=INK)
    if tail == "bl":
        for k, rr in ((0, 13), (1, 8)):
            cx = x0 + 44 + k * 30; cy = y1 + 16 + k * 22
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=fill, outline=outline, width=3)

def ease(t):                      # smootherstep
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6 - 15) + 10)

def text(d, xy, s, f, fill=INK, anchor="la"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)

def fade(img, a):
    if a >= 1.0: return img
    return Image.blend(Image.new("RGB", img.size, BG), img, max(0.0, a))

# ---------------------------------------------------------------- episodes
def load_episodes(n=9):
    meta = {}
    for ln in open(JSONL, errors="ignore"):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("correct"): meta[str(r.get("qid"))] = r
    eps = []
    # val452 leads: "What is at the end of the hallway?" is unambiguous and its
    # scene mesh (scene0046_00) is the one pre-rendered for the camera panel
    for qid in ["val452", "val748", "val560", "val136", "val91", "val219",
                "val813", "val93", "val678", "val1", "val331", "val221"]:
        hit = glob.glob(os.path.join(VIEWS, f"ScanNet_{qid}_*"))
        if not hit or qid not in meta: continue
        fr = sorted(glob.glob(os.path.join(hit[0], "turn_*.png")))
        if len(fr) < 4: continue
        q = meta[qid].get("question", "").strip()
        steps = []
        poses = []
        for t in meta[qid].get("turns", []):
            pz = t.get("pose") or {}
            if pz.get("eye") and pz.get("at"): poses.append((pz["eye"], pz["at"]))
            txt = str(t.get("model_text", "")).split("</think>")[0]
            txt = re.sub(r"\s+", " ", txt.replace("<think>", "")).strip()
            steps.append({"think": txt, "act": str(t.get("action_or_coords", "")).strip()})
        eps.append({"q": q, "a": str(meta[qid].get("final_answer", "")).strip(),
                    "steps": steps, "poses": poses,
                    "frames": [Image.open(p).convert("RGB") for p in fr[:8]]})
        if len(eps) == n: break
    return eps

def grid_cell(ep, k, size):
    """one episode tile showing frame k, with its question underneath"""
    cw, ch = size
    ih = int(cw * 480 / 640)
    tile = Image.new("RGB", (cw, ch), CARD)
    im = ep["frames"][k % len(ep["frames"])].resize((cw, ih), Image.LANCZOS)
    tile.paste(im, (0, 0))
    d = ImageDraw.Draw(tile)
    f = font(17)
    q = ep["q"]
    if len(q) > 46: q = q[:44] + "..."
    d.text((10, ih + 9), q, font=f, fill=INK)
    d.text((10, ih + 32), f"step {k % len(ep['frames'])}", font=font(15), fill=SOFT)
    d.rectangle([0, 0, cw - 1, ch - 1], outline=LINE)
    return tile

# ---------------------------------------------------------------- scenes
def sc_title(t, dur, eps):
    """big title over a dim, slowly cycling trace grid"""
    img = Image.new("RGB", (W, H), BG)
    cols, rows, pad = 4, 3, 10
    cw = (W - pad * (cols + 1)) // cols
    ch = int(cw * 480 / 640)
    oy = (H - (ch * rows + pad * (rows - 1))) // 2
    for r in range(rows):
        for c in range(cols):
            ep = eps[(r * cols + c) % len(eps)]
            k = int(t * 2.2) + r + c
            im = ep["frames"][k % len(ep["frames"])].resize((cw, ch), Image.LANCZOS)
            img.paste(im, (pad + c * (cw + pad), oy + r * (ch + pad)))
    img = Image.blend(Image.new("RGB", (W, H), BG), img, 0.18)
    d = ImageDraw.Draw(img)
    big = font(190, True)
    for off, col in (((7, 8), (188, 198, 214)), ((3, 4), (150, 163, 184)), ((0, 0), INK)):
        d.text((W // 2 + off[0], H // 2 - 70 + off[1]), "Act3D", font=big, fill=col, anchor="mm")
    if t > 0.7:
        b = ease((t - 0.7) / 0.8)
        sub = font(50, True)
        shade = tuple(int(BG[i] + (200 - BG[i]) * b) for i in range(3))
        d.text((W // 2 + 3, H // 2 + 63), "Learning to Think and Explore in 3D World",
               font=sub, fill=shade, anchor="mm")
        d.text((W // 2, H // 2 + 60), "Learning to Think and Explore in 3D World",
               font=sub, fill=tuple(int(BG[i] + (ACCENT[i] - BG[i]) * b) for i in range(3)), anchor="mm")
    if t > 1.5:
        b = ease((t - 1.5) / 0.8)
        au = font(34, True)
        col = tuple(int(BG[i] + (INK[i] - BG[i]) * b) for i in range(3))
        col2 = tuple(int(BG[i] + (SOFT[i] - BG[i]) * b) for i in range(3))
        d.text((W // 2, H // 2 + 138), "Yijiang Li†      Bingyang Wang†      Nuno Vasconcelos",
               font=au, fill=col, anchor="mm")
        d.text((W // 2, H // 2 + 186), "UC San Diego   ·   Georgia Institute of Technology",
               font=font(27, True), fill=col2, anchor="mm")
        d.text((W // 2, H // 2 + 230), "† equal contribution",
               font=font(24), fill=col2, anchor="mm")
    return fade(img, min(1.0, t / 0.5))

_BEV = {}
def bev_assets():
    """the real mesh render plus the camera that produced it"""
    if not _BEV:
        here = os.path.dirname(os.path.abspath(__file__))
        _BEV["img"] = Image.open(os.path.join(here, "static/images/scene_bev.png")).convert("RGB")
        _BEV["cam"] = json.load(open(os.path.join(here, "static/images/scene_bev.json")))
    return _BEV["img"], _BEV["cam"]

def _world_to_img(P, cam):
    """bev_render.Camera.project, re-implemented without numpy"""
    aa, ee = math.radians(cam["azim"]), math.radians(cam["elev"])
    x = P[0] * math.cos(aa) + P[1] * math.sin(aa)
    y = -P[0] * math.sin(aa) + P[1] * math.cos(aa)
    u = x
    v = y * math.sin(ee) + P[2] * math.cos(ee)
    u0, u1, v0, v1 = cam["extent"]
    ppm = cam["px_per_m"]
    return ((u - u0) * ppm, (v1 - v) * ppm)

def scene_panel(ep, k, local, per, box):
    """right half: the real 3D scene render with the camera flying its actual path"""
    x0, y0, x1, y1 = box
    pw, ph = x1 - x0, y1 - y0
    panel = Image.new("RGB", (pw, ph), (250, 251, 253))
    bev, cam = bev_assets()
    sc = min((pw - 40) / bev.width, (ph - 86) / bev.height)
    bw, bh = int(bev.width * sc), int(bev.height * sc)
    ox = (pw - bw) // 2
    oy = 18 + ((ph - 86) - bh) // 2          # centre in the band above the caption
    panel.paste(bev.resize((bw, bh), Image.LANCZOS), (ox, oy))
    d = ImageDraw.Draw(panel)
    rrect(d, [0, 0, pw - 1, ph - 1], 26, outline=LINE, width=3)

    def P(w):                                   # world -> panel pixels
        u, v = _world_to_img(w, cam)
        return (ox + u * sc, oy + v * sc)

    poses = ep["poses"]
    if not poses:
        return panel
    pts = [P(e) for e, _ in poses[:k + 1]]

    # frustums already visited, drawn faintly
    ov = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    for i, (e, at) in enumerate(poses[:k + 1]):
        cp, ap = P(e), P(at)
        vx, vy = ap[0] - cp[0], ap[1] - cp[1]
        L = max(1e-6, math.hypot(vx, vy)); vx, vy = vx / L, vy / L
        nx, ny = -vy, vx
        far, half = 108, 46
        od.polygon([cp, (cp[0] + vx * far + nx * half, cp[1] + vy * far + ny * half),
                    (cp[0] + vx * far - nx * half, cp[1] + vy * far - ny * half)],
                   fill=BLUE + (58,), outline=BLUE + (150,))
    panel = Image.alpha_composite(panel.convert("RGBA"), ov).convert("RGB")
    d = ImageDraw.Draw(panel)

    if len(pts) > 1:
        d.line(pts, fill=(70, 110, 200), width=5, joint="curve")

    # the live camera, interpolated towards the next pose
    f = min(1.0, max(0.0, (local - 0.45 * per) / (0.5 * per))) if k + 1 < len(poses) else 0.0
    e0, a0 = poses[k]
    e1, a1 = poses[min(k + 1, len(poses) - 1)]
    eye = [e0[i] + (e1[i] - e0[i]) * f for i in range(3)]
    at = [a0[i] + (a1[i] - a0[i]) * f for i in range(3)]
    cp, ap = P(eye), P(at)

    glow = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    vx, vy = ap[0] - cp[0], ap[1] - cp[1]
    L = max(1e-6, math.hypot(vx, vy)); vx, vy = vx / L, vy / L
    nx, ny = -vy, vx
    gd.polygon([cp, (cp[0] + vx * 150 + nx * 64, cp[1] + vy * 150 + ny * 64),
                (cp[0] + vx * 150 - nx * 64, cp[1] + vy * 150 - ny * 64)], fill=GLOW + (95,))
    for rr, al in ((54, 40), (36, 70), (22, 120)):
        gd.ellipse([cp[0] - rr, cp[1] - rr, cp[0] + rr, cp[1] + rr], fill=GLOW + (al,))
    panel = Image.alpha_composite(panel.convert("RGBA"), glow).convert("RGB")
    d = ImageDraw.Draw(panel)

    # numbered badges for every pose so far
    for i, q in enumerate(pts):
        r = 17 if i < len(pts) - 1 else 21
        col = (255, 255, 255) if i < len(pts) - 1 else GLOW
        d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=col, outline=ACCENT, width=4)
        d.text((q[0], q[1] + 1), str(i), font=font(22 if i < len(pts) - 1 else 25, True),
               fill=ACCENT, anchor="mm")

    d.text((24, ph - 44), "scene mesh  ·  the camera poses it chose", font=font(25), fill=SOFT)
    return panel

def wrap(s_, f, width, d):
    out, line = [], ""
    for w_ in s_.split():
        t = (line + " " + w_).strip()
        if d.textlength(t, font=f) <= width: line = t
        else: out.append(line); line = w_
    if line: out.append(line)
    return out

def sc_walk(t, dur, eps):
    """robot + its thought bubble, the chosen view, the move in the middle, the scene right"""
    ep = eps[0]
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2, 50), ep["q"], font=font(46, True), fill=INK, anchor="mm")
    d.text((W // 2, 100), "one initial view, then every move is the model's own",
           font=font(26), fill=SOFT, anchor="mm")

    nst = min(len(ep["frames"]), len(ep["steps"]), max(1, len(ep["poses"])))
    per = (dur - 1.4) / nst
    k = min(nst - 1, max(0, int(t / per)))
    local = t - k * per

    # ---- robot and a fixed-size bubble; only the words fade in each step
    rx, ry = 86, 196
    robot(img, rx, ry, sz=1.0)
    BUB_H = 40 + 4 * 36
    bx2, by2, bw2 = rx + 164, ry - 34, 620
    rrect(d, [bx2, by2, bx2 + bw2, by2 + BUB_H], 26, fill=(238, 243, 255), outline=BLUE, width=3)
    for kk, rr in ((0, 14), (1, 9)):
        cx = bx2 - 24 - kk * 28; cy = by2 + BUB_H * 0.62 + kk * 20
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(238, 243, 255), outline=BLUE, width=3)
    d.text((rx + 6, ry + 116), "thinking", font=font(23, True), fill=BLUE)
    tf = ease(local / 0.4)                                  # the words flash in, the bubble does not
    if tf > 0:
        tc = tuple(int(238 + (INK[i] - 238) * tf) for i in range(3))
        for i, ln in enumerate(wrap(ep["steps"][k]["think"], font(27), bw2 - 56, d)[:4]):
            d.text((bx2 + 26, by2 + 18 + i * 36), ln, font=font(27), fill=tc)

    # ---- the view it is looking at
    iw = 640; ih = int(iw * 480 / 640)
    ix, iy = 92, 432
    im = ep["frames"][k].resize((iw, ih), Image.LANCZOS)
    msk = Image.new("L", (iw, ih), 0)
    ImageDraw.Draw(msk).rounded_rectangle([0, 0, iw - 1, ih - 1], 22, fill=255)
    img.paste(im, (ix, iy), msk)
    rrect(d, [ix, iy, ix + iw, iy + ih], 22, outline=LINE, width=3)
    d.text((ix + 6, iy - 36), f"step {k}  of  {nst - 1}", font=font(26, True), fill=SOFT)

    # ---- the move, centred in the gap between the two halves
    if k < nst - 1:
        f2 = font(31, True)
        acts = [st["act"].replace("+", "  ×") for st in ep["steps"][:nst - 1]]
        wd = int(max(d.textlength(a2, font=f2) for a2 in acts)) + 56   # fixed: no resizing
        cx = 862
        cy = iy + ih // 2 - 35
        d.text((cx, cy - 34), "move", font=font(24, True), fill=TEAL, anchor="mm")
        rrect(d, [cx - wd // 2, cy, cx + wd // 2, cy + 70], 35,
              fill=(224, 243, 246), outline=TEAL, width=3)
        af = ease((local - 0.35) / 0.4)                                 # only the action text flashes
        if af > 0:
            ac = tuple(int(224 + (16 - 224) * af), ) if False else (
                int(224 + (16 - 224) * af), int(243 + (86 - 243) * af), int(246 + (99 - 246) * af))
            d.text((cx, cy + 35), acts[k], font=f2, fill=ac, anchor="mm")

    # ---- the answer, beneath the view
    if k == nst - 1 and local > 0.5:
        txt = f'answer:  {ep["a"]}'
        wd = int(d.textlength(txt, font=font(38, True))) + 84
        rrect(d, [ix, iy + ih + 28, ix + wd, iy + ih + 112], 42,
              fill=(222, 242, 229), outline=GREEN, width=3)
        d.text((ix + wd // 2, iy + ih + 70), txt, font=font(38, True), fill=(26, 94, 62), anchor="mm")

    # ---- the scene, right half
    panel = scene_panel(ep, k, local, per, (1000, 150, 1870, 1010))
    img.paste(panel, (1000, 150))
    return fade(img, min(1.0, t / 0.4) * min(1.0, (dur - t) / 0.4))

def _strip(frames, n, tw, th, kind):
    """small preview of an input modality, built from real frames / the mesh render"""
    gap = 8
    strip = Image.new("RGB", (n * tw + (n - 1) * gap, th), BG)
    for i in range(n):
        im = frames[i % len(frames)].resize((tw, th), Image.LANCZOS)
        if kind == "depth":
            im = im.convert("L").point(lambda v: 40 + v * 0.75).convert("RGB")
        strip.paste(im, (i * (tw + gap), 0))
    d = ImageDraw.Draw(strip)
    if kind == "video":                       # film sprockets
        for i in range(0, strip.width, 22):
            d.rectangle([i + 5, 2, i + 13, 8], fill=(255, 255, 255))
            d.rectangle([i + 5, th - 9, i + 13, th - 3], fill=(255, 255, 255))
    if kind == "depth":                       # a couple of 3D boxes
        for bx in ((10, 14, 62, 60), (tw + 26, 22, tw + 78, 66)):
            d.rectangle(bx, outline=GREEN, width=3)
    return strip

def sc_prior(t, dur, eps):
    """prior work is handed the evidence; Act3D is handed one view and a way to act"""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2, 46), "Prior 3D vision–language models assume complete evidence as input",
           font=font(42, True), fill=INK, anchor="mm")

    bev, _ = bev_assets()
    frames = eps[1]["frames"] if len(eps) > 1 else eps[0]["frames"]
    items = [("point clouds & meshes", GREEN, "mesh"),
             ("posed multi-view images", BLUE, "multi"),
             ("video sequences", PURPLE, "video"),
             ("depth, poses, 3D boxes", TEAL, "depth")]

    # ---- top half: one tidy row per modality, all feeding a centred VLM
    lx, lw, rh, gap = 96, 300, 96, 20
    ry0, bh = 108, 80
    px, pw, th = 430, 376, 80                 # preview column
    ax = px + pw + 34                         # where the arrows start
    rows_mid = ry0 + (4 * rh + 3 * gap) // 2
    vlm = (1210, rows_mid - 92, 1700, rows_mid + 92)

    for i, (lab, col, kind) in enumerate(items):
        a_ = ease((t - 0.2 * i) / 0.5)
        if a_ <= 0: continue
        cy = ry0 + i * (rh + gap) + rh // 2
        box = Image.new("RGB", (lw, bh), CARD)
        dd = ImageDraw.Draw(box)
        rrect(dd, [1, 1, lw - 2, bh - 2], 24, fill=CARD, outline=col, width=4)
        dd.text((lw // 2, bh // 2), lab, font=font(25), fill=INK, anchor="mm")
        if a_ < 1: box = Image.blend(Image.new("RGB", box.size, BG), box, a_)
        img.paste(box, (lx, cy - bh // 2))

        if t > 0.2 * i + 0.22:
            b2 = ease((t - 0.2 * i - 0.22) / 0.45)
            if kind == "mesh":
                mw = int(bev.width * th / bev.height)
                strip = Image.new("RGB", (pw, th), BG)
                strip.paste(bev.resize((mw, th), Image.LANCZOS), ((pw - mw) // 2, 0))
            else:
                n = 4 if kind == "video" else 3
                tw = (pw - (n - 1) * 8) // n
                strip = _strip(frames, n, tw, th, kind)
            if b2 < 1: strip = Image.blend(Image.new("RGB", strip.size, BG), strip, b2)
            sx, sy = px, cy - th // 2
            msk = Image.new("L", strip.size, 0)
            ImageDraw.Draw(msk).rounded_rectangle([0, 0, strip.width - 1, strip.height - 1], 12, fill=255)
            img.paste(strip, (sx, sy), msk)
            rrect(d, [sx - 3, sy - 3, sx + strip.width + 2, sy + strip.height + 2], 14,
                  outline=col, width=2)
            if t > 0.2 * i + 0.55:
                p_ = ease((t - 0.2 * i - 0.55) / 0.55)
                tx, ty = vlm[0] - 24, rows_mid
                arrow(d, (ax, cy), (ax + (tx - ax) * p_, cy + (ty - cy) * p_), col,
                      w=9, bow=0.07, head=24)

    if t > 1.1:
        a_ = ease((t - 1.1) / 0.5)
        box = Image.new("RGB", (vlm[2] - vlm[0], vlm[3] - vlm[1]), (235, 240, 252))
        dd = ImageDraw.Draw(box)
        rrect(dd, [1, 1, box.width - 2, box.height - 2], 34, fill=(235, 240, 252),
              outline=ACCENT, width=5)
        dd.text((box.width // 2, box.height // 2), "VLM", font=font(62, True), fill=ACCENT, anchor="mm")
        if a_ < 1: box = Image.blend(Image.new("RGB", box.size, BG), box, a_)
        img.paste(box, (vlm[0], vlm[1]))

    # ---- bottom half: our setup, one centred row
    if t > 2.6:
        a_ = ease((t - 2.6) / 0.6)
        d.line([140, 582, 1780, 582], fill=LINE, width=3)
        c = tuple(int(BG[i] + (INK[i] - BG[i]) * a_) for i in range(3))
        d.text((W // 2, 644), "Act3D treats 3D as an interactable environment",
               font=font(44, True), fill=c, anchor="mm")
        c2 = tuple(int(BG[i] + (SOFT[i] - BG[i]) * a_) for i in range(3))
        d.text((W // 2, 698), "preserving the native visual encoding of VLMs",
               font=font(29), fill=c2, anchor="mm")
    if t > 3.1:
        ep = eps[2] if len(eps) > 2 else eps[0]   # not the episode the demo walks through
        fr = ep["frames"][:max(3, len(ep["frames"]) // 2)]   # the opening half of the trace
        nst = len(fr) - 1                                   # moves we can act out
        pos = (t - 3.1) / 1.35
        k, u = int(pos) % nst, pos - int(pos)
        iw = 300; ih = int(iw * 480 / 640)
        im = fr[k].resize((iw, ih), Image.LANCZOS)
        a_ = ease((t - 3.1) / 0.5)
        if a_ < 1: im = Image.blend(Image.new("RGB", im.size, BG), im, a_)
        ix, iy = 300, 752
        msk = Image.new("L", im.size, 0)
        ImageDraw.Draw(msk).rounded_rectangle([0, 0, iw - 1, ih - 1], 20, fill=255)
        img.paste(im, (ix, iy), msk)
        rrect(d, [ix, iy, ix + iw, iy + ih], 20, outline=LINE, width=3)
        if a_ > 0.6:                                        # acting out that same move
            walker(d, ix + 60, iy + ih - 18, 64, ep["steps"], k, ease(u))

        bx2, by2 = 790, iy + 24
        arrow(d, (ix + iw + 22, iy + ih // 2), (bx2 - 26, by2 + 60), ACCENT, w=9, bow=0.14, head=24)
        bw3, bh3 = 1390 - bx2, 140
        rrect(d, [bx2, by2, bx2 + bw3, by2 + bh3], 28, fill=(238, 243, 255), outline=BLUE, width=3)
        words = ["observe", "reason", "act", "observe"]
        f3 = font(30)
        cw3 = chain_w(d, words, f3)
        chain(d, (bx2 + (bw3 - cw3) / 2, by2 + bh3 / 2 - 46), words, f3, INK, BLUE)
        d.text((bx2 + bw3 / 2, by2 + bh3 / 2 + 30), "until the evidence is enough",
               font=font(27), fill=SOFT, anchor="mm")
        for kk, rr in ((0, 13), (1, 8)):
            cx = bx2 + 46 + kk * 30; cy = by2 + bh3 + 16 + kk * 22
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(238, 243, 255), outline=BLUE, width=3)

        if t > 4.2:                                   # the human-like framing pops in at the right
            pop = ease((t - 4.2) / 0.45)
            tag = ["a single image, one at a time,", "just like humans"]
            tf2 = font(28, True)
            tw2 = int(max(d.textlength(l, font=tf2) for l in tag)) + 52
            tx = bx2 + bw3 + 40
            cyc = by2 + bh3 / 2 + (1 - pop) * 26
            hh = 36 + 34 * pop
            rrect(d, [tx, cyc - hh, tx + tw2, cyc + hh], min(34, int(hh * 0.9)),
                  fill=(222, 242, 229), outline=GREEN, width=4)
            if pop > 0.5:
                for i, l in enumerate(tag):
                    d.text((tx + tw2 // 2, cyc - 20 + i * 40), l, font=tf2,
                           fill=(26, 94, 62), anchor="mm")
    return fade(img, min(1.0, t / 0.4) * min(1.0, (dur - t) / 0.4))

def sc_data(t, dur, eps):
    """corpora and the two training stages — no ablation content"""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2, 62), "What we train on", font=font(44, True), fill=INK, anchor="mm")
    # two animated stacked bars
    bars = [("5,712 supervised trajectories", 250,
             [("ScanQA", 3000, BLUE), ("MP3D", 2000, ORANGE), ("ESI", 712, GREEN)]),
            ("51,543 questions for RL", 430,
             [("ScanQA", 25563, BLUE), ("MP3D", 9855, ORANGE),
              ("curated ScanNet", 8591, GREEN), ("curated ARKit", 7534, PURPLE)])]
    for bi, (lab, y, segs) in enumerate(bars):
        a = ease((t - 0.3 - 0.5 * bi) / 1.0)
        tot = sum(n for _, n, _ in segs)
        x = 220; full = 1480
        d.text((220, y - 46), lab, font=font(31, True), fill=INK)
        for nm, n, col in segs:
            w = int(full * n / tot * a)
            if w > 0:
                rrect(d, [x, y, x + w, y + 62], 16, fill=col)
                lab2 = f"{nm}  {n:,}"
                fnt = font(22)
                if w > d.textlength(lab2, font=fnt) + 28:
                    d.text((x + 14, y + 31), lab2, font=fnt, fill=(255, 255, 255), anchor="lm")
                elif w > d.textlength(nm, font=fnt) + 28:
                    d.text((x + 14, y + 31), nm, font=fnt, fill=(255, 255, 255), anchor="lm")
            x += int(full * n / tot * a)
    if t > 2.0:
        a = ease((t - 2.0) / 0.6)
        c = tuple(int(BG[i] + (SOFT[i] - BG[i]) * a) for i in range(3))
        d.text((220, 548), "questions kept only if they stay well posed from a single first view",
               font=font(27), fill=c)
    if 2.6 < t < 7.6:
        a = ease((t - 2.6) / 0.7) * min(1.0, (7.6 - t) / 0.6)
        y = 660
        stages = [("trajectory-supervised initialization", "establishes reason–act–observe", 220),
                  ("outcome-driven RL (GRPO)", "reward: did the final answer come out right", 1020)]
        for nm, sub, x in stages:
            box = Image.new("RGB", (640, 150), CARD)
            dd = ImageDraw.Draw(box)
            rrect(dd, [1, 1, 638, 148], 28, fill=CARD, outline=ACCENT, width=4)
            dd.text((320, 55), nm, font=font(29, True), fill=INK, anchor="mm")
            dd.text((320, 100), sub, font=font(23), fill=SOFT, anchor="mm")
            if a < 1: box = Image.blend(Image.new("RGB", box.size, BG), box, max(0.0, a))
            img.paste(box, (x, y))
        if t > 3.1 and a > 0.55:
            arrow(d, (875, y + 75), (1005, y + 75), ACCENT, w=12, bow=0.0, head=30)

    if t > 7.9:
        a = ease((t - 7.9) / 0.8)
        c = tuple(int(BG[i] + (INK[i] - BG[i]) * a) for i in range(3))
        d.text((W // 2, 648), "the whole reward", font=font(34, True), fill=c, anchor="mm")
        f, fs = mfont(46), mfont(27)
        x = 232
        x = draw_runs(d, x, 724, [("R(τ) = ", None)], f, fs, c)
        t1a = x
        x = draw_runs(d, x, 724, [("λ", "judge"), ("R", "judge"), (" + ", None),
                                  ("λ", "EM"), ("R", "EM")], f, fs, c)
        t1b = x
        x = draw_runs(d, x, 724, [("  +  ", None)], f, fs, c)
        t2a = x
        x = draw_runs(d, x, 724, [("λ", "int"), ("R", "judge"), ("c(τ)", None)], f, fs, c)
        t2b = x
        x = draw_runs(d, x, 724, [("  −  ", None)], f, fs, c)
        t3a = x
        x = draw_runs(d, x, 724, [("λ", "fmt"), ("R", "fmt")], f, fs, c)
        t3b = x
        if t > 8.5:
            b = ease((t - 8.5) / 0.7)
            for (xa, xb, col, lab) in ((t1a, t1b, BLUE, "answer correctness"),
                                       (t2a, t2b, TEAL, "interaction reward"),
                                       (t3a, t3b, PURPLE, "output format")):
                cc = tuple(int(BG[i] + (col[i] - BG[i]) * b) for i in range(3))
                brace(d, xa, xb, 806, cc, lab, font(26, True))
        if t > 9.6:
            b = ease((t - 9.6) / 0.7)
            c2 = tuple(int(BG[i] + (SOFT[i] - BG[i]) * b) for i in range(3))
            d.text((W // 2, 930), "the interaction term is paid only when the answer is right, and saturates at twelve",
                   font=font(28), fill=c2, anchor="mm")
            c3 = tuple(int(BG[i] + (ACCENT[i] - BG[i]) * b) for i in range(3))
            d.text((W // 2, 990), "no supervision of viewpoints, actions, or trajectories",
                   font=font(31, True), fill=c3, anchor="mm")
    return fade(img, min(1.0, t / 0.4) * min(1.0, (dur - t) / 0.4))

def sc_result(t, dur, eps):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2, 80), "Six 3D benchmarks, one interaction protocol",
           font=font(44, True), fill=INK, anchor="mm")
    rows = [("Qwen3.5-9B  (base)", 31.2, (205, 213, 225)),
            ("InternVL3.5-38B", 43.9, (205, 213, 225)),
            ("Ross3D", 50.7, (205, 213, 225)),
            ("Gemini 3.6 Flash", 52.2, (160, 174, 192)),
            ("Act3D  (ours)", 53.2, ACCENT)]
    y0, bh, gap = 220, 86, 34
    for i, (nm, v, col) in enumerate(rows):
        a = ease((t - 0.25 * i) / 0.8)
        y = y0 + i * (bh + gap)
        d.text((560, y + bh // 2), nm, font=font(30, True if i == 4 else False),
               fill=INK if i == 4 else SOFT, anchor="rm")
        w = int(1080 * (v / 60.0) * a)
        rrect(d, [600, y, 600 + max(w, 18), y + bh], 18, fill=col)
        if a > 0.6:
            d.text((600 + w + 18, y + bh // 2), f"{v:.1f}", font=font(32, True),
                   fill=INK if i == 4 else SOFT, anchor="lm")
    if t > 2.0:
        a = ease((t - 2.0) / 0.7)
        c = tuple(int(BG[i] + (INK[i] - BG[i]) * a) for i in range(3))
        d.text((W // 2, 870), "+22.0 over its base model, from a single starting view",
               font=font(36, True), fill=c, anchor="mm")
        c2 = tuple(int(BG[i] + (SOFT[i] - BG[i]) * a) for i in range(3))
        d.text((W // 2, 930), "gains grow with how much exploration the question needs: +0.6  to  +16.3",
               font=font(30), fill=c2, anchor="mm")
    return fade(img, min(1.0, t / 0.4) * min(1.0, (dur - t) / 0.4))

def sc_end(t, dur, eps):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2, H // 2 - 120), "Act3D", font=font(130, True), fill=INK, anchor="mm")
    d.text((W // 2, H // 2 - 10), "Learning to Think and Explore in 3D World",
           font=font(42), fill=ACCENT, anchor="mm")
    if t > 0.5:
        a = ease((t - 0.5) / 0.7)
        c = tuple(int(BG[i] + (SOFT[i] - BG[i]) * a) for i in range(3))
        d.text((W // 2, H // 2 + 95), "arXiv:2610.21783", font=font(36, True, mono=True), fill=c, anchor="mm")
        d.text((W // 2, H // 2 + 155), "williamium3000.github.io/act3d", font=font(32, mono=True), fill=c, anchor="mm")
    return fade(img, min(1.0, t / 0.5))

SCENES = [(sc_title, 5.0), (sc_prior, 15.0), (sc_walk, 19.0),
          (sc_data, 15.0), (sc_result, 9.0), (sc_end, 5.0)]

def main():
    eps = load_episodes(9)
    if len(eps) < 6:
        sys.exit("not enough episodes with frames")
    total = sum(d for _, d in SCENES)
    print(f"episodes: {len(eps)}  duration: {total:.0f}s  frames: {int(total*FPS)}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-preset", "medium", "-crf", "21", "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    n = 0
    for fn, dur in SCENES:
        for i in range(int(dur * FPS)):
            p.stdin.write(fn(i / FPS, dur, eps).tobytes())
            n += 1
            if n % 150 == 0: print(f"  {n}/{int(total*FPS)}", flush=True)
    p.stdin.close(); p.wait()
    print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB")

if __name__ == "__main__":
    main()
