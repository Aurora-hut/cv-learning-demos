# -*- coding: utf-8 -*-
"""
第六个 Demo —— 灯条配对：不用神经网络，纯几何认出"装甲板"
------------------------------------------------------------------
前面 demo3/4/5 都在用 YOLO。但 RM 赛场上真正自瞄的第一步不是"神经网络认车"，
而是**找灯条**：装甲板上那两条发光的竖条，形状特征极其稳定
（细长、竖直、成对、间距约等于灯条高度）。用 20 行 OpenCV 几何就能抓得很稳，
CPU 上 1~2 ms，比跑网络快一个数量级。

完整流程（RM 战队视觉组的标准 pipeline 的最小版）：
  ① 灰度 + 高斯模糊
  ② 亮度阈值 → 二值图（灯条是画面里最亮的东西）
  ③ findContours + minAreaRect → 每个亮块的"长宽比 / 填充率 / 倾角"
  ④ 形状筛选：留下细长且接近竖直的 → 灯条候选
  ⑤ **两两配对**：高度相近 + 中心高度接近 + 倾角一致 + 水平间距合理 → 装甲板
  ⑥ 贪心去重：一个灯条只能属于一块装甲板

运行：
  A. 无参数：自动生成 1 张合成场景图跑一遍（不用摄像头也能看效果）
  B. 图片：  python demo6_armor.py --image 你的图.jpg
  C. 摄像头：python demo6_armor.py
  D. 视频：  python demo6_armor.py --video xxx.mp4
  E. 量化自测：python demo6_armor.py --selftest
  F. 调阈值：--thr 170（灯不亮就调低，误检多就调高）
  G. 存中间二值图：--debug

注意：detect_armors() 会自动从 200 往下降阈值重试，所以 --thr 只在你想锁定单一阈值时才用。
"""

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# 关键参数（想调效果基本只动这几个）
# ----------------------------------------------------------------------------
DEFAULT_THRS = (190, 165, 140, 115, 90)   # 亮度阈值，从高往低试，取整体最优（不是第一个出结果的）
MIN_ASPECT = 2.0        # 长宽比下限（灯条长/宽，真实约 4~8；方块干扰就靠它挡）
MAX_ASPECT = 14.0       # 长宽比上限
MIN_FILL = 0.42         # 填充率下限（轮廓面积 / 外接矩形面积，实心灯条接近 1）
MAX_TILT = 42.0         # 倾角绝对值上限（度），0 = 完全竖直
COLOR_DIFF = 45          # 通道差阈值：B-max(R,G) > 45 判为蓝色灯条，红反之
MIN_AREA_RATIO = 0.00004  # 面积下限（占全图比例）
MAX_AREA_RATIO = 0.06     # 面积上限


# ----------------------------------------------------------------------------
# 中文路径安全读写（OpenCV 5 打不开含中文的绝对路径）
# ----------------------------------------------------------------------------
def imread_u(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(path)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("decode failed: " + path)
    return img


def imwrite_u(path, img):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


# ----------------------------------------------------------------------------
# 第 1 步：从二值图里挑出"可能是灯条"的亮块
# ----------------------------------------------------------------------------
def find_bars(mask, min_fill=MIN_FILL):
    H, W = mask.shape[:2]
    total = float(H * W)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bars = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < total * MIN_AREA_RATIO or area > total * MAX_AREA_RATIO:
            continue

        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), _ = rect
        if rw < 2.0 or rh < 2.0:
            continue
        # 归一化：保证 rh 是长边
        if rw > rh:
            rw, rh = rh, rw
        aspect = rh / rw
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        # 填充率：细长但空心的一律丢掉（排除"两条细线围成的大框"）
        fill = area / (rw * rh + 1e-6)
        if fill < min_fill:
            continue

        # 用 PCA 主轴定方向。
        # 为什么不用 minAreaRect 的 angle：它的取值约定在 4.3 / 4.5 / 5.0 之间反复改过，
        # 用 boxPoints 反推长边方向也会被 w/h 的先后顺序坑到（-90 度就翻车）。
        # 轮廓点云的协方差主轴只跟形状本身有关，换版本也不会变。
        pts = c.reshape(-1, 2).astype(np.float32)
        if len(pts) < 5:
            continue
        d = pts - pts.mean(axis=0)
        cov = (d.T @ d) / max(1, len(d) - 1)
        eig_val, eig_vec = np.linalg.eigh(cov)
        axis = eig_vec[:, int(np.argmax(eig_val))]        # 单位向量，指向长边
        # 0 度 = 竖直向上，正值 = 上端向右倒
        tilt = math.degrees(math.atan2(float(axis[0]), float(-axis[1])))
        tilt = (tilt + 90.0) % 180.0 - 90.0
        if abs(tilt) > MAX_TILT:
            continue

        # 灯条的真实长宽（rh 是长边）
        bars.append({
            "cx": float(cx), "cy": float(cy),
            "w": float(rw), "h": float(rh),
            "tilt": float(tilt),
            "area": float(area),
            "pts": cv2.boxPoints(rect).astype(np.float32),
        })
    return bars


# ----------------------------------------------------------------------------
# 第 2 步：两两配对成装甲板
# ----------------------------------------------------------------------------
def pair_bars(bars):
    cands = []
    n = len(bars)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = bars[i], bars[j]
            ha, hb = a["h"], b["h"]
            hmean = (ha + hb) * 0.5

            # ① 高度不能差太多（同一条装甲板上的灯条一样长）
            h_ratio = min(ha, hb) / max(ha, hb)
            if h_ratio < 0.6:
                continue

            # ② 两个灯条的中心高度接近
            dy = abs(a["cy"] - b["cy"])
            if dy > 0.50 * hmean:
                continue

            # ③ 倾角一致（透视下同一块装甲板的两条灯条倾斜方向相同）
            d_tilt = abs(a["tilt"] - b["tilt"])
            if d_tilt > 10.0:
                continue

            # ④ 水平间距：RM 小装甲板约 1.0~1.3 倍灯条高，大装甲板约 1.8~2.3 倍
            gap = abs(a["cx"] - b["cx"])
            ratio = gap / hmean
            if ratio < 0.70 or ratio > 6.0:
                continue
            # 两条灯条不能横向重叠
            if gap < 0.45 * (a["w"] + b["w"]):
                continue

            # ⑤ 综合评分（越像标准装甲板越高）
            score = h_ratio
            score *= max(0.0, 1.0 - dy / (0.5 * hmean))
            score *= max(0.0, 1.0 - d_tilt / 10.0)
            score *= 1.0 / (1.0 + abs(ratio - 1.5))
            # ⑥ 大灯条更可信（远处的小亮块多半是噪点/装饰灯）
            score *= min(1.0, math.sqrt(hmean / 45.0))
            cands.append((score, i, j))

    # 贪心：分数高的先占位，一个灯条只归一块装甲板
    cands.sort(key=lambda t: -t[0])
    used = set()
    armors = []
    for score, i, j in cands:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        armors.append(_build_armor(bars, i, j, score))
    return armors


def _build_armor(bars, i, j, score):
    a, b = bars[i], bars[j]
    left, right = (a, b) if a["cx"] <= b["cx"] else (b, a)
    all_pts = np.vstack([a["pts"], b["pts"]])
    rect = cv2.minAreaRect(all_pts)
    (rcx, rcy), (rw, rh), _ = rect
    # 保证 w 是横向的长边（装甲板整体是横的长方形）
    if rw < rh:
        rcx, rcy, rw, rh = rcx, rcy, rh, rw
    return {
        "cx": float((left["cx"] + right["cx"]) * 0.5),
        "cy": float((a["cy"] + b["cy"]) * 0.5),
        "w": float(rw), "h": float(rh),
        "score": float(score),
        "left": left, "right": right,
        "box": cv2.boxPoints(((rcx, rcy), (rw, rh), 0.0)).astype(np.int32),
        "rect": (rcx, rcy, rw, rh),
    }


# ----------------------------------------------------------------------------
# 对外接口：一帧图 → 装甲板列表
# ----------------------------------------------------------------------------
def color_masks(blurred, thr_col=COLOR_DIFF):
    """blurred 传已做一次 3x3 高斯模糊的 BGR 图（省一次滤波）。

    按"通道差"找纯色发光体：蓝灯条 B 远大于 R、G；红灯条反过来。

    为什么不能只用灰度？BGR(255,95,45) 的蓝色灯条算成灰度只有 137，
    比白墙还暗，灰度阈值根本抓不到它，只会抓到灯条正中那一小条过曝白芯。
    通道差对"纯色 + 高饱和"特别敏感，而场地背景几乎不会这么纯。
    """
    bi = blurred[:, :, 0].astype(np.int16)
    gi = blurred[:, :, 1].astype(np.int16)
    ri = blurred[:, :, 2].astype(np.int16)
    blue = (bi - np.maximum(gi, ri)) > thr_col
    red = (ri - np.maximum(gi, bi)) > thr_col
    return ((blue | red).astype(np.uint8)) * 255


def detect_armors(frame, thr=None, min_fill=MIN_FILL, thr_col=COLOR_DIFF):
    """返回 (armors, bars, mask, used_thr)。thr=None 时自动从高到低试。"""
    blur = cv2.GaussianBlur(frame, (3, 3), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    cmask = color_masks(blur, thr_col)

    thrs = [thr] if thr is not None else list(DEFAULT_THRS)
    k_open = np.ones((3, 3), np.uint8)
    k_close = np.ones((3, 7), np.uint8)   # 竖向闭运算：把被噪声切断的灯条接回去

    best = ([], [], np.zeros_like(gray), thrs[0])
    best_score = -1e9
    for t in thrs:
        _, bright = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(bright, cmask)      # 白光 + 彩色光，两种灯条一起收
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        bars = find_bars(mask, min_fill=min_fill)
        armors = pair_bars(bars) if len(bars) >= 2 else []

        # 打分：装甲板越多越好，剩下没配对的"孤灯条"每个扣分
        # ——"第一个出结果的阈值就收手"是错的：低阈值往往能多捞出一块暗一点的装甲板
        orphan = len(bars) - 2 * len(armors)
        sc = sum(a["score"] for a in armors) - 0.25 * orphan
        if sc > best_score:
            best_score = sc
            best = (armors, bars, mask, t)
        if armors and orphan <= 1 and len(armors) >= len(best[0]):
            break                                  # 灯条基本都成对了，不必再降阈值
    return best


# ----------------------------------------------------------------------------
# 画图
# ----------------------------------------------------------------------------
def draw(frame, armors, bars, used_thr=None):
    out = frame.copy()

    for b in bars:
        cv2.polylines(out, [b["pts"].astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)

    for k, ar in enumerate(armors):
        cv2.polylines(out, [ar["box"]], True, (0, 0, 255), 2, cv2.LINE_AA)
        c = (int(ar["cx"]), int(ar["cy"]))
        cv2.drawMarker(out, c, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(out, "A%d %.2f" % (k, ar["score"]), (c[0] + 10, c[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    txt = "bars=%d  armors=%d  thr=%s" % (len(bars), len(armors), used_thr)
    cv2.putText(out, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 1, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------
# 合成场景（没有真实素材时也能验证算法）
# ----------------------------------------------------------------------------
def _draw_bar(img, cx, cy, w, h, tilt, color):
    """画一根带倾角的灯条：外圈是原色，往里一层层混白，模拟发光过曝。

    注意必须是"一层层盖不透明色"，不能用 addWeighted 反复和 img 混合——
    那样第二条灯条会继承第一条的衰减，越画越暗，最后测出来的是数据问题而不是算法问题。
    """
    th = math.radians(tilt)
    ux, uy = math.sin(th), -math.cos(th)      # 长边方向（向上）
    vx, vy = math.cos(th), math.sin(th)       # 短边方向

    # (缩放, 混白比例)  从外到内
    for scale, white in ((1.00, 0.00), (0.74, 0.32), (0.54, 0.62), (0.34, 0.90)):
        hh = h * 0.5 * scale
        ww = w * 0.5 * scale
        c = tuple(int(ch * (1.0 - white) + 255.0 * white) for ch in color)
        pts = [(cx + ux * hh * sy + vx * ww * sx,
                cy + uy * hh * sy + vy * ww * sx)
               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        cv2.fillPoly(img, [np.array(pts, np.int32)], c)


def make_scene(W=960, H=540, seed=0, n_armor=1, clutter=True, noise=6.0):
    """生成一张合成场景图 + 真值列表 [(cx, cy, w, h), ...]。

    画图顺序很重要：背景 → 干扰物 → 装甲板。
    灯条永远在最上层，因为现实里灯条就是画面中最亮、最不该被挡住的东西；
    如果让干扰物盖住灯条，测出来的"漏检"是数据造的假 bug，不是算法的。
    """
    rng = np.random.default_rng(seed)

    # ---- 背景：深灰渐变 + 网格（模拟场地/机房）----
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img = (34.0 + 18.0 * yy) * np.ones((H, W, 3), np.float32)
    for x in range(0, W, 80):
        cv2.line(img, (x, 0), (x, H), (52, 52, 52), 1)
    for y in range(0, H, 60):
        cv2.line(img, (0, y), (W, y), (52, 52, 52), 1)

    # ---- 干扰物：亮度刻意压在灯条之下（<190），用来测"会不会误检"----
    if clutter:
        for _ in range(int(rng.integers(4, 10))):
            rw = int(rng.integers(12, 90))
            rh = int(rng.integers(12, 90))
            rx = int(rng.integers(10, max(11, W - rw - 10)))
            ry = int(rng.integers(10, max(11, H - rh - 10)))
            v = int(rng.integers(88, 152))
            cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (v, v, v), -1)

    # ---- 装甲板：拒绝采样，保证彼此不重叠 ----
    gt = []
    placed = []
    for _ in range(n_armor):
        for _try in range(60):
            h = float(rng.integers(46, 120))
            w = h * float(rng.uniform(0.15, 0.23))
            gap = h * float(rng.uniform(0.95, 2.2))
            tilt = float(rng.uniform(-26, 26))
            aw, ah = gap + 2 * w, h
            cx = float(rng.uniform(0.16, 0.84) * W)
            cy = float(rng.uniform(0.20, 0.80) * H)
            if cx - aw < 8 or cx + aw > W - 8:
                continue
            if cy - ah < 8 or cy + ah > H - 8:
                continue
            ok = True
            r_new = 0.5 * math.hypot(aw, ah)      # 用外接圆判重叠，倾斜 20 多度也不会误判
            for (px, py, pr) in placed:
                if math.hypot(cx - px, cy - py) < r_new + pr + 22:
                    ok = False
                    break
            if ok:
                placed.append((cx, cy, r_new))
                break
        else:
            continue   # 实在放不下就少画一块

        color = (255, 95, 45) if rng.random() < 0.5 else (60, 65, 255)  # BGR 蓝 / 红

        # 装甲板底板（暗）
        th = math.radians(tilt)
        ux, uy, vx, vy = math.sin(th), -math.cos(th), math.cos(th), math.sin(th)
        bh, bw = h * 0.5, (gap + 2 * w) * 0.5
        base = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            base.append((cx + ux * bh * sy + vx * bw * sx,
                         cy + uy * bh * sy + vy * bw * sx))
        cv2.fillPoly(img, [np.array(base, np.int32)], (46, 46, 48))

        for s in (-1, 1):
            bcx = cx + s * (gap * 0.5 + w * 0.5)
            _draw_bar(img, bcx, cy, w, h, tilt, color)
        gt.append((cx, cy, gap + 2 * w, h))

    img = np.asarray(img, np.uint8)
    if noise > 0:
        n = rng.normal(0, noise, img.shape)
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return img, gt


# ----------------------------------------------------------------------------
# 量化自测：检出率 / 中心误差 / 误检 / 耗时
# ----------------------------------------------------------------------------
def run_selftest(n=30, save_dir=None, quiet=False):
    hit = tot = 0
    errs = []
    false_pos = 0
    times = []

    for k in range(n):
        rng = np.random.default_rng(1000 + k)
        n_armor = int(rng.integers(1, 4))
        img, gt = make_scene(seed=k, n_armor=n_armor)

        t0 = time.perf_counter()
        armors, bars, mask, thr = detect_armors(img)
        times.append((time.perf_counter() - t0) * 1000.0)

        used = set()
        matched = 0
        for (gx, gy, gw, gh) in gt:
            best, bi = None, -1
            for idx, ar in enumerate(armors):
                if idx in used:
                    continue
                d = math.hypot(ar["cx"] - gx, ar["cy"] - gy)
                if best is None or d < best:
                    best, bi = d, idx
            tot += 1
            # 中心误差容忍 0.35 倍灯条高度
            if bi >= 0 and best <= 0.35 * gh:
                used.add(bi)
                hit += 1
                matched += 1
                errs.append(best)
        false_pos += max(0, len(armors) - matched)

        if save_dir and k < 4:
            os.makedirs(save_dir, exist_ok=True)
            imwrite_u(os.path.join(save_dir, "selftest_%02d.jpg" % k),
                      draw(img, armors, bars, thr))

    r = {
        "cases": n,
        "gt": tot,
        "hit": hit,
        "recall": hit / tot if tot else 0.0,
        "err_med": float(np.median(errs)) if errs else float("nan"),
        "err_p90": float(np.percentile(errs, 90)) if errs else float("nan"),
        "false_pos": false_pos,
        "ms_med": float(np.median(times)),
        "ms_p90": float(np.percentile(times, 90)),
    }
    if not quiet:
        print("=" * 58)
        print("合成场景自测  %d 个场景 / %d 块装甲板" % (n, tot))
        print("  检出率      %.1f%%  (%d/%d)" % (r["recall"] * 100, hit, tot))
        print("  中心误差    median %.2f px   p90 %.2f px" % (r["err_med"], r["err_p90"]))
        print("  误检        多出 %d 个框" % false_pos)
        print("  单帧耗时    median %.2f ms   p90 %.2f ms  (960x540)" % (r["ms_med"], r["ms_p90"]))
        print("=" * 58)
    return r


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="单张图片")
    ap.add_argument("--video", default=None, help="视频文件")
    ap.add_argument("--thr", type=int, default=None, help="锁定亮度阈值（默认自动 200→120）")
    ap.add_argument("--min-fill", type=float, default=MIN_FILL, help="填充率下限")
    ap.add_argument("--selftest", action="store_true", help="跑合成场景量化自测")
    ap.add_argument("--selftest-n", type=int, default=30)
    ap.add_argument("--debug", action="store_true", help="额外保存二值图 demo6_mask.jpg")
    ap.add_argument("--no-show", action="store_true", help="不弹窗口（只存文件）")
    ap.add_argument("--out", default=None, help="输出文件名")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    # ---- 自测 ----
    if args.selftest:
        run_selftest(args.selftest_n, save_dir=os.path.join(BASE, "docs", "demo6"))
        return

    # ---- 无参数：合成演示 ----
    if not args.image and not args.video and not args.selftest:
        print("没有给图片/视频，先用合成场景演示一遍（不会开摄像头）")
        img, gt = make_scene(seed=7, n_armor=2)
        armors, bars, mask, thr = detect_armors(img)
        out = draw(img, armors, bars, thr)
        imwrite_u(os.path.join(BASE, "demo6_scene.jpg"), img)   # 存原图，方便重复测
        p = os.path.join(BASE, "demo6_out.jpg")
        imwrite_u(p, out)
        if args.debug:
            imwrite_u(os.path.join(BASE, "demo6_mask.jpg"), mask)
        print("  灯条候选 %d 条，装甲板 %d 块（阈值 %d）" % (len(bars), len(armors), thr))
        print("  真值 %d 块" % len(gt))
        print("  已保存 -> demo6_out.jpg")
        if not args.no_show:
            _show(out, "demo6 - synthetic")
        return

    # ---- 单张图片 ----
    if args.image:
        img = imread_u(args.image)
        armors, bars, mask, thr = detect_armors(img, thr=args.thr, min_fill=args.min_fill)
        out = draw(img, armors, bars, thr)
        p = args.out or os.path.join(BASE, "demo6_out.jpg")
        imwrite_u(p, out)
        if args.debug:
            imwrite_u(os.path.join(BASE, "demo6_mask.jpg"), mask)
        print("  灯条候选 %d 条，装甲板 %d 块（阈值 %d）" % (len(bars), len(armors), thr))
        for k, ar in enumerate(armors):
            print("    #%d 中心(%.0f,%.0f) %.0fx%.0f  score=%.2f"
                  % (k, ar["cx"], ar["cy"], ar["w"], ar["h"], ar["score"]))
        print("  已保存 -> %s" % os.path.basename(p))
        if not args.no_show:
            _show(out, "demo6 - " + os.path.basename(args.image))
        return

    # ---- 视频 / 摄像头 ----
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        print("打开失败：", args.video or ("camera %d" % args.camera))
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    writer = None
    if args.video:
        out_p = args.out or os.path.join(BASE, "demo6_out.mp4")
        writer = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    n, t_sum, t_max = 0, 0.0, 0.0
    show = not args.no_show
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        armors, bars, mask, thr = detect_armors(frame, thr=args.thr, min_fill=args.min_fill)
        dt = (time.perf_counter() - t0) * 1000.0
        n += 1
        t_sum += dt
        t_max = max(t_max, dt)

        out = draw(frame, armors, bars, thr)
        cv2.putText(out, "%.1f ms" % dt, (10, out.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        if writer is not None:
            writer.write(out)

        if show:
            cv2.imshow("demo6 - armor", out)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord("q"):
                break
            if k == ord("s"):
                imwrite_u(os.path.join(BASE, "demo6_snapshot.jpg"), out)
                print("  已存快照 demo6_snapshot.jpg")

    cap.release()
    if writer is not None:
        writer.release()
        print("  已保存 -> %s" % os.path.basename(args.out or "demo6_out.mp4"))
    if show:
        cv2.destroyAllWindows()
    if n:
        print("  共 %d 帧，平均 %.2f ms/帧（%.1f FPS），最慢 %.2f ms"
              % (n, t_sum / n, 1000.0 / (t_sum / n), t_max))


def _show(img, title):
    try:
        cv2.imshow(title, img)
        print("  按任意键关闭窗口…")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("  （当前环境没有图形界面，已跳过显示）")


if __name__ == "__main__":
    main()
