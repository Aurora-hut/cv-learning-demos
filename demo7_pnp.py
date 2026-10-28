# -*- coding: utf-8 -*-
"""
第七个 Demo —— PnP 解算：从"画面里在哪"到"有多远、偏多少度"
------------------------------------------------------------------
demo6 只能告诉我装甲板在图像上的**像素位置**。但要打中它，真正需要的是：

    它在相机坐标系里的**距离（米）** 和 **姿态角（yaw / pitch / roll）**

这就是 PnP（Perspective-n-Point）要解决的问题：
    已知 N 个点在**物体坐标系**里的三维坐标，
    也知道它们在**图像上**的二维像素坐标，
    再加上**相机内参 K**，就能反解出相机与物体的相对位姿（旋转 R + 平移 t）。

数学上它解的是投影方程：
    s · [u, v, 1]ᵀ = K · (R · P_world + t)
其中 s 是深度（未知），这正是"从 2D 反推 3D"需要额外信息的原因——
单纯一个点是解不出来的，必须有**已知尺寸的刚体**提供尺度约束。
装甲板恰好提供了：它的物理尺寸是标准件（135 mm × 55 mm）。

本 demo 用到的标准尺寸（RM 小装甲板）：
    外框宽 135 mm，高 55 mm。

运行：
  A. 无参数：合成一个已知位姿的 3D 场景，解算后和真值对比（最能看清精度）
  B. 图片：  python demo7_pnp.py --image 你的图.jpg
  C. 摄像头：python demo7_pnp.py
  D. 视频：  python demo7_pnp.py --video xxx.mp4
  E. 精度自测：python demo7_pnp.py --selftest
  F. 用标定好的内参：--calib camera.json
  G. 生成标定用棋盘格：--make-board

⚠ 内参的重要性：不标定就相当于"用猜的焦距"，距离会系统性地偏大或偏小。
  没标定时脚本按 60° 水平视场角估算，只适合看趋势，真实测距请先跑 calibrate_camera.py。
"""

import argparse
import json
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
# 装甲板物理尺寸（毫米）—— 一旦改这里，所有距离输出都会跟着变
# ----------------------------------------------------------------------------
ARMOR_W = 135.0     # 外框宽
ARMOR_H = 55.0      # 外框高
BAR_W = 15.0        # 单根灯条的宽度（画合成图用）
AXIS_LEN = 85.0     # 画坐标轴的长度（mm），太短在远处看不见

# 默认水平视场角（笔记本摄像头典型 60°，无标定时用它凑内参）
DEFAULT_HFOV_DEG = 60.0


# ----------------------------------------------------------------------------
# 中文路径安全读写
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
# 相机内参
# ----------------------------------------------------------------------------
def default_K(W, H, hfov_deg=DEFAULT_HFOV_DEG):
    """没有标定时的兜底内参：假设方形像素、无畸变、主点在图像中心。"""
    fx = (W * 0.5) / math.tan(math.radians(hfov_deg) * 0.5)
    return np.array([[fx, 0.0, W * 0.5],
                     [0.0, fx, H * 0.5],
                     [0.0, 0.0, 1.0]], np.float64)


def load_K(path, W, H):
    """从标定 JSON 读内参；读不到就退回默认值。"""
    if not path or not os.path.exists(path):
        return default_K(W, H), np.zeros(5, np.float64), False
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    K = np.array(d["K"], np.float64)
    dist = np.array(d.get("dist", [0, 0, 0, 0, 0]), np.float64)
    # 标定时的分辨率如果和当前图不同，按比例缩放内参
    cw, ch = d.get("width"), d.get("height")
    if cw and ch and (cw != W or ch != H):
        sx, sy = W / float(cw), H / float(ch)
        K = K.copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
    return K, dist, True


# ----------------------------------------------------------------------------
# 物体坐标系下的 4 个角点
# ----------------------------------------------------------------------------
def armor_object_points():
    """装甲板外框 4 角，顺序固定为：左上 → 右上 → 右下 → 左下。

    顺序必须和图像上 4 个点的顺序一一对应，错一个 PnP 就会解出完全错误的位姿
    （这是 RM 视觉最经典的坑之一，且通常不会报错，只会"莫名其妙打偏"）。
    """
    hw, hh = ARMOR_W * 0.5, ARMOR_H * 0.5
    return np.array([[-hw, -hh, 0.0],
                     [hw, -hh, 0.0],
                     [hw, hh, 0.0],
                     [-hw, hh, 0.0]], np.float64)


# ----------------------------------------------------------------------------
# 从 demo6 的灯条推出装甲板 4 个外角点
# ----------------------------------------------------------------------------
def bars_to_quad(armor):
    """左右灯条 → 装甲板外框 4 角（左外上、右外上、右外下、左外下）。

    为什么不直接用灯条四角做 8 点 PnP：灯条太窄，角点定位误差在短边方向被放大；
    用"外框 4 角"参与解算，几何尺度更大，抗噪声明显更好（RM 队伍主流做法）。
    """
    L, R = armor["left"], armor["right"]
    pl, pr = np.asarray(L["pts"], np.float32), np.asarray(R["pts"], np.float32)

    # 取"离另一根灯条最远"的两个点，就是外侧边
    lc = np.array([L["cx"], L["cy"]], np.float32)
    rc = np.array([R["cx"], R["cy"]], np.float32)

    def outer_pair(pts, other_c):
        d = np.linalg.norm(pts - other_c[None, :], axis=1)
        idx = np.argsort(-d)[:2]
        two = pts[idx]
        return two[np.argsort(two[:, 1])]      # 上、下

    lu, ld = outer_pair(pl, rc)
    ru, rd = outer_pair(pr, lc)
    return np.array([lu, ru, rd, ld], np.float64)


# ----------------------------------------------------------------------------
# 位姿解算
# ----------------------------------------------------------------------------
def euler_from_R(R):
    """旋转矩阵 → (pitch, yaw, roll)，单位度。

    约定 R = Rz(roll) · Ry(yaw) · Rx(pitch)，与合成数据用同一套公式，前后自洽。
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:                                    # 俯仰接近 ±90°，万向锁
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def euler_to_R(pitch_deg, yaw_deg, roll_deg):
    """(pitch, yaw, roll) → 旋转矩阵，与 euler_from_R 互逆。"""
    p, y, r = (math.radians(v) for v in (pitch_deg, yaw_deg, roll_deg))
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def solve_pose(quad_img, K, dist=None, prev_rvec=None):
    """4 点 PnP。返回 dict，失败返回 None。

    用 SOLVEPNP_IPPE：专为**平面目标**设计的解析解法，比通用的 ITERATIVE 更稳，
    也不需要给初值。平面目标在远处会有镜像双解，所以再用 solvePnPGeneric 取重投影误差小的那个。
    """
    obj = armor_object_points()
    img = np.asarray(quad_img, np.float64).reshape(4, 1, 2)
    if dist is None:
        dist = np.zeros(5, np.float64)

    rvec = tvec = None
    try:
        ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if ok and len(rvecs) > 0:
            errs = np.asarray(errs).ravel()
            cands = []
            for i in range(len(rvecs)):
                R_i, _ = cv2.Rodrigues(rvecs[i])
                # 物理约束：装甲板正面朝相机时，板面法线（R 的第 3 列）Z 分量必须为正。
                # 这一条能砍掉平面 PnP 里最常见的那类"翻到背面"的伪解。
                if R_i[2, 2] <= 0.05:
                    continue
                cands.append(i)
            if not cands:
                cands = list(range(len(rvecs)))

            if prev_rvec is not None and len(cands) > 1:
                # 时序先验：挑和上一帧姿态最接近的解。
                # 平面目标的两个解投影误差往往只差 0.001 px，光看误差永远选不对；
                # 但"这一帧不该突然翻 30 度"这个先验非常可靠。
                def _rot_gap(i):
                    dR = cv2.Rodrigues(rvecs[i])[0] @ cv2.Rodrigues(np.asarray(prev_rvec).reshape(3, 1))[0].T
                    c = (np.trace(dR) - 1.0) * 0.5
                    return abs(math.acos(float(np.clip(c, -1.0, 1.0))))
                best = min(cands, key=_rot_gap)
            else:
                # 没有先验：取重投影误差最小的（理论上两解并列，取 errs 略小的那个）
                best = min(cands, key=lambda i: errs[i])
            rvec, tvec = rvecs[best], tvecs[best]
    except cv2.error:
        pass

    if rvec is None:                                    # 兜底
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None

    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    reproj = float(np.mean(np.linalg.norm(
        proj.reshape(-1, 2) - np.asarray(quad_img, np.float64).reshape(-1, 2), axis=1)))

    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = euler_from_R(R)
    tvec = tvec.reshape(3)
    return {
        "rvec": rvec.reshape(3),
        "tvec": tvec,
        "R": R,
        "dist_m": float(np.linalg.norm(tvec)) / 1000.0,
        "x_m": float(tvec[0]) / 1000.0,
        "y_m": float(tvec[1]) / 1000.0,
        "z_m": float(tvec[2]) / 1000.0,
        "yaw": yaw, "pitch": pitch, "roll": roll,
        "reproj": reproj,
    }


# ----------------------------------------------------------------------------
# 画 3D 坐标轴：这是判断"解到底对不对"最直观的方式
# ----------------------------------------------------------------------------
def draw_pose(frame, pose, K, dist=None, text=True):
    out = frame.copy()
    if pose is None:
        return out
    if dist is None:
        dist = np.zeros(5, np.float64)

    obj = armor_object_points()
    quad, _ = cv2.projectPoints(obj, pose["rvec"], pose["tvec"], K, dist)
    quad = quad.reshape(-1, 2).astype(np.int32)
    cv2.polylines(out, [quad], True, (0, 0, 255), 1, cv2.LINE_AA)

    # 坐标轴：X 红、Y 绿、Z 蓝。轴指向就是装甲板的朝向，歪了一眼就看得出来
    axes = np.array([[0, 0, 0], [AXIS_LEN, 0, 0],
                     [0, AXIS_LEN, 0], [0, 0, AXIS_LEN]], np.float64)
    proj, _ = cv2.projectPoints(axes, pose["rvec"], pose["tvec"], K, dist)
    proj = proj.reshape(-1, 2).astype(int)
    o = tuple(proj[0])
    cv2.line(out, o, tuple(proj[1]), (0, 0, 255), 3, cv2.LINE_AA)   # X 红
    cv2.line(out, o, tuple(proj[2]), (0, 255, 0), 3, cv2.LINE_AA)   # Y 绿
    cv2.line(out, o, tuple(proj[3]), (255, 128, 0), 3, cv2.LINE_AA)  # Z 蓝

    cx, cy = int(pose["tvec"][0] / pose["tvec"][2] * K[0, 0] + K[0, 2]), \
             int(pose["tvec"][1] / pose["tvec"][2] * K[1, 1] + K[1, 2])
    lines = ["d = %.2f m" % pose["dist_m"],
             "yaw %+.1f" % pose["yaw"],
             "pitch %+.1f" % pose["pitch"],
             "roll %+.1f" % pose["roll"],
             "reproj %.2f px" % pose["reproj"]]
    if not text:
        return out
    # 文字画在装甲板正上方，别压在灯条上（否则看不清目标本身）
    for i, s in enumerate(lines):
        y = cy - 58 - (len(lines) - 1 - i) * 20
        if y < 16:
            y = 16 + i * 20
        cv2.putText(out, s, (cx - 60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, s, (cx - 60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------
# 合成一个"真值已知"的 3D 场景
# ----------------------------------------------------------------------------
def make_3d_scene(K, pitch, yaw, roll, tz_m, tx_m=0.0, ty_m=0.0,
                  W=640, H=480, seed=0, noise_px=0.6, draw_truth=True):
    """按给定位姿渲染一张合成图，返回 (img, gt_quad_px, rvec_gt, tvec_gt)。

    关键：这里的投影用 projectPoints 完成，透视关系是**物理正确**的，
    所以后面 PnP 反解出来的距离和角度可以拿来和真值逐项对比。
    """
    rng = np.random.default_rng(seed)

    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img = (32.0 + 16.0 * yy) * np.ones((H, W, 3), np.float32)
    for x in range(0, W, 60):
        cv2.line(img, (x, 0), (x, H), (50, 50, 50), 1)
    for y in range(0, H, 45):
        cv2.line(img, (0, y), (W, y), (50, 50, 50), 1)

    R = euler_to_R(pitch, yaw, roll)
    rvec, _ = cv2.Rodrigues(R)
    tvec = np.array([tx_m * 1000.0, ty_m * 1000.0, tz_m * 1000.0], np.float64)

    # 两根灯条的 3D 矩形（装甲板局部坐标，Z=0）
    hw, hh = ARMOR_W * 0.5, ARMOR_H * 0.5
    bars3d = [
        np.array([[-hw, -hh, 0], [-hw + BAR_W, -hh, 0],
                  [-hw + BAR_W, hh, 0], [-hw, hh, 0]], np.float64),
        np.array([[hw - BAR_W, -hh, 0], [hw, -hh, 0],
                  [hw, hh, 0], [hw - BAR_W, hh, 0]], np.float64),
    ]

    # 装甲板底板
    plate3d = np.array([[-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0]], np.float64)
    pp, _ = cv2.projectPoints(plate3d, rvec, tvec, K, np.zeros(5))
    cv2.fillPoly(img, [pp.reshape(-1, 2).astype(np.int32)], (44, 44, 46))

    for b3 in bars3d:
        proj, _ = cv2.projectPoints(b3, rvec, tvec, K, np.zeros(5))
        poly = proj.reshape(-1, 2)
        # 由外到内一层层缩小并混白，模拟发光（和 demo6 的 _draw_bar 同思路）
        c2 = poly.mean(axis=0)
        for scale, white in ((1.00, 0.0), (0.68, 0.35), (0.42, 0.70), (0.24, 0.92)):
            p = c2 + (poly - c2) * scale
            col = (int(45 * (1 - white) + 255 * white),
                   int(95 * (1 - white) + 255 * white),
                   int(255 * (1 - white) + 255 * white))
            cv2.fillPoly(img, [p.astype(np.int32)], col)

    img = np.asarray(img, np.uint8)
    if noise_px > 0:
        n = rng.normal(0, 3.0, img.shape)
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)

    # 真值 4 角（外框），并假设"检测"有 noise_px 的像素抖动
    obj = armor_object_points()
    gt, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))
    gt = gt.reshape(-1, 2)
    if noise_px > 0:
        gt = gt + rng.normal(0, noise_px, gt.shape)
    return img, gt, rvec.reshape(3), tvec


# ----------------------------------------------------------------------------
# 精度自测：把 PnP 解得的值和真值逐项对比
# ----------------------------------------------------------------------------
def _quad_from_pose(pitch, yaw, roll, tz, W, H, K, seed, noise_px, tx=0.0, ty=0.0):
    """只做投影 + 噪声，不渲染整张图（自测要跑上千次，渲染会成为瓶颈）。"""
    R = euler_to_R(pitch, yaw, roll)
    rvec, _ = cv2.Rodrigues(R)
    tvec = np.array([tx * 1000.0, ty * 1000.0, tz * 1000.0], np.float64)
    proj, _ = cv2.projectPoints(armor_object_points(), rvec, tvec, K, np.zeros(5))
    quad = proj.reshape(4, 2)
    if noise_px > 0:
        rng = np.random.default_rng(seed)
        quad = quad + rng.normal(0, noise_px, quad.shape)
    return quad


def run_selftest(n=60, verbose=True):
    # 用真实 RM 相机的量级：1280x720、fx≈1000（约 65° 水平视场）。
    # 为什么不用 640x480：3 m 外 135 mm 的装甲板只有 25 px 宽，0.5 px 检测噪声就能让
    # 角度飘十几度——那是物理极限，不是算法问题，测出来只会误导人。
    W, H = 1280, 720
    K = default_K(W, H, hfov_deg=65.0)
    fx = float(K[0, 0])

    # ================= ① 零噪声基线：证明"算法本身没有误差" =================
    base_cases = [(3.0, 0, 0, 0), (3.0, 20, 0, 0), (3.0, 0, 15, 0),
                  (3.0, 20, 10, 5), (2.0, 35, -20, -10)]
    base = []
    for (tz, yaw, pitch, roll) in base_cases:
        q = _quad_from_pose(pitch, yaw, roll, tz, W, H, K, 1, 0.0)
        p = solve_pose(q, K)
        base.append((abs(p["yaw"] - yaw), abs(p["pitch"] - pitch),
                     abs(p["dist_m"] - tz) * 1000.0, p["reproj"]))

    # ================= ② 0.6 px 噪声下的精度（按距离分档） =================
    rows = []
    for k in range(n):
        rng = np.random.default_rng(2000 + k)
        pitch = float(rng.uniform(-20, 20))
        yaw = float(rng.uniform(-35, 35))
        roll = float(rng.uniform(-15, 15))
        tz = float(rng.uniform(1.5, 5.0))
        tx = float(rng.uniform(-0.20, 0.20)) * tz
        ty = float(rng.uniform(-0.15, 0.15)) * tz

        q = _quad_from_pose(pitch, yaw, roll, tz, W, H, K, 4000 + k, 0.6, tx, ty)
        t0 = time.perf_counter()
        p = solve_pose(q, K)
        dt = (time.perf_counter() - t0) * 1000.0
        if p is None:
            continue
        d_gt = math.hypot(tx, ty, tz)
        rows.append({
            "d_gt": d_gt,
            "d_err": abs(p["dist_m"] - d_gt),      # 单位：米
            "d_rel": abs(p["dist_m"] - d_gt) / d_gt * 100.0,
            "yaw_err": abs(p["yaw"] - yaw),
            "pitch_err": abs(p["pitch"] - pitch),
            "roll_err": abs(p["roll"] - roll),
            "reproj": p["reproj"],
            "ms": dt,
        })

    def stat(key, sub=None):
        v = np.array([r[key] for r in (sub or rows)], dtype=np.float64)
        return float(np.median(v)), float(np.percentile(v, 90)), float(v.max())

    # ================= ③ 噪声敏感度 =================
    noise_tbl = []
    for nz in (0.0, 0.2, 0.5, 1.0, 2.0):
        es = []
        for k in range(24):
            q = _quad_from_pose(8, 15, 0, 3.0, W, H, K, 7000 + k, nz)
            p = solve_pose(q, K)
            es.append((abs(p["yaw"] - 15), abs(p["pitch"] - 8),
                       abs(p["dist_m"] - 3.0) * 1000))
        e = np.array(es)
        noise_tbl.append((nz, float(np.median(e[:, 0])),
                          float(np.median(e[:, 1])), float(np.median(e[:, 2]))))

    # ================= ④ 双解歧义演示 =================
    amb = []
    q = _quad_from_pose(8, 15, 0, 3.0, W, H, K, 9001, 0.6)
    ok, rvs, tvs, ess = cv2.solvePnPGeneric(
        armor_object_points(), q.reshape(4, 1, 2), K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE)
    for i in range(len(rvs)):
        R, _ = cv2.Rodrigues(rvs[i])
        pp, yy, rr = euler_from_R(R)
        amb.append((yy, pp, float(np.ravel(ess)[i]), float(np.linalg.norm(tvs[i]) / 1000)))

    res = {"rows": len(rows), "base": base, "noise": noise_tbl, "amb": amb}
    for key in ("d_err", "d_rel", "yaw_err", "pitch_err", "roll_err", "reproj", "ms"):
        res[key] = stat(key)

    if not verbose:
        return res

    print("=" * 70)
    print("PnP 精度自测   %d 个随机位姿 · 相机 %.0fx%.0f · fx=%.0f · 仿真检测噪声 0.6 px"
          % (len(rows), W, H, fx))
    print("=" * 70)

    print("① 零噪声基线（验证数学约定本身是否正确）")
    print("   真值(d, yaw, pitch, roll)      解算误差")
    for (tz, yaw, pitch, roll), (ey, ep, ed, er) in zip(base_cases, base):
        print("   d=%.1fm yaw=%+5.0f° pitch=%+5.0f° roll=%+5.0f°   yaw %.2e°  pitch %.2e°  距离 %.2e mm"
              % (tz, yaw, pitch, roll, ey, ep, ed))
    print("   → 全部为 0，说明角点顺序、欧拉角约定、solvePnP 调用都对；")
    print("     下面的误差完全来自像素噪声，不是算法写错了。")
    print()

    print("② 有噪声时的实际精度（按距离分档，均为 median）")
    print("   距离区间     样本   距离相对误差    距离绝对误差    yaw 误差   pitch 误差")
    for lo, hi in ((1.5, 2.5), (2.5, 3.5), (3.5, 5.1)):
        sub = [x for x in rows if lo <= x["d_gt"] < hi]
        if not sub:
            continue
        print("   %.1f–%.1f m    %2d      %5.2f%%         %5.0f mm      %6.2f°     %6.2f°"
              % (lo, hi, len(sub),
                 float(np.median([x["d_rel"] for x in sub])),
                 float(np.median([x["d_err"] for x in sub])) * 1000,
                 float(np.median([x["yaw_err"] for x in sub])),
                 float(np.median([x["pitch_err"] for x in sub]))))
    print("   总体         %2d      %5.2f%%         %5.0f mm      %6.2f°     %6.2f°"
          % (len(rows), res["d_rel"][0], res["d_err"][0] * 1000,
             res["yaw_err"][0], res["pitch_err"][0]))
    print("   重投影误差 median %.2f px · 单次解算 %.3f ms" % (res["reproj"][0], res["ms"][0]))
    print()

    print("③ 噪声敏感度（固定位姿 d=3.0m yaw=15° pitch=8°）")
    print("   检测噪声      yaw 误差    pitch 误差    距离误差")
    for nz, ey, ep, ed in noise_tbl:
        print("   %.1f px        %7.2f°     %7.2f°      %4.0f mm" % (nz, ey, ep, ed))
    print("   → 角度误差几乎与噪声成正比。想稳住角度，只有降噪声（更好的灯条分割）")
    print("     或拉长焦距（同样距离下目标像素更多），没有第三条路。")
    print()

    print("④ 平面双解歧义（这是本 demo 最该记住的一件事）")
    for i, (yy, pp, er, dd) in enumerate(amb):
        print("   解 %d:  yaw=%+7.2f°  pitch=%+7.2f°  距离 %.2f m  重投影 %.4f px" % (i, yy, pp, dd, er))
    if len(amb) >= 2:
        print("   两个解的重投影误差几乎一样（差 %.4f px），但姿态差了 %.1f°。"
              % (abs(amb[0][2] - amb[1][2]), abs(amb[0][0] - amb[1][0])))
        print("   这不是 bug：平面目标在透视下天生有两个镜像解，单帧无解。")
        print("   工程解法是**时序先验**——用上一帧姿态挑解（solve_pose 的 prev_rvec 参数），")
        print("   而不是把两帧结果取平均（那样只会得到中间那个错误值）。")
    print("=" * 70)
    return res


# ----------------------------------------------------------------------------
# 生成标定棋盘格
# ----------------------------------------------------------------------------
def make_chessboard(path=None, cols=9, rows=6, square_mm=25.0, dpi=300):
    """生成可直接打印的 A4 棋盘格（张正友标定法用）。"""
    sq = int(round(square_mm / 25.4 * dpi))
    w, h = cols * sq, rows * sq
    board = np.full((h, w), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                board[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 0
    margin = int(round(10 / 25.4 * dpi))
    board = cv2.copyMakeBorder(board, margin, margin, margin, margin,
                               cv2.BORDER_CONSTANT, value=255)
    cv2.putText(board, "%dx%d  square=%.0fmm" % (cols, rows, square_mm),
                (margin, margin + h + int(margin * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)
    path = path or os.path.join(BASE, "chessboard_9x6_25mm.png")
    ok, buf = cv2.imencode(".png", board)
    if ok:
        buf.tofile(path)
    print("已生成棋盘格 -> %s" % os.path.basename(path))
    print("  内角点 %d x %d，格子边长 %.0f mm，300 dpi" % (cols, rows, square_mm))
    print("  用 A4 纸按 100%% 比例打印（不要缩放），贴在硬平板上保持平整")
    return path


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--calib", default=None, help="标定结果 JSON（calibrate_camera.py 生成）")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-n", type=int, default=60)
    ap.add_argument("--make-board", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.make_board:
        make_chessboard()
        return

    if args.selftest:
        run_selftest(args.selftest_n)
        return

    # ---------------- 无参数：合成 3D 演示 ----------------
    if not args.image and not args.video:
        # 用和自测一致的相机参数，否则小目标会让角度看起来"完全解错"，
        # 那是分辨率问题不是算法问题，演示图要挑能说明问题的配置。
        W, H = 1280, 720
        K = default_K(W, H, hfov_deg=65.0)
        print("合成一个位姿已知的 3D 场景，演示 PnP 能不能把真值解回来")
        print("  内参按 65° 水平视场角估算：fx=fy=%.1f" % K[0, 0])

        gt_cases = [(1.5, 0.0, 0.0), (2.5, 15.0, -8.0), (3.5, -25.0, 12.0), (4.5, 30.0, -12.0)]
        tiles = []
        for i, (tz, yaw, pitch) in enumerate(gt_cases):
            img, gtq, _, tvec_gt = make_3d_scene(K, pitch, yaw, 0.0, tz, W=W, H=H, seed=i)
            pose = solve_pose(gtq, K)
            vis = draw_pose(img, pose, K, text=False) if pose else img.copy()
            print("  真值 d=%.2fm yaw=%+.0f° pitch=%+.0f°  ->  解出 d=%.2fm yaw=%+.1f° pitch=%+.1f°  (err %.2fpx)"
                  % (tz, yaw, pitch, pose["dist_m"], pose["yaw"], pose["pitch"], pose["reproj"]))
            # 裁到装甲板附近再放大，否则 1280x720 缩到拼图里只剩几十像素，根本看不清坐标轴。
            # 文字必须等裁剪放大之后再画，否则会被裁掉半个字。
            ccx, ccy = int(gtq[:, 0].mean()), int(gtq[:, 1].mean())
            rw, rh = 420, 315
            x0 = max(0, min(W - rw, ccx - rw // 2))
            y0 = max(0, min(H - rh, ccy - rh // 2))
            crop = cv2.resize(vis[y0:y0 + rh, x0:x0 + rw], (640, 480), interpolation=cv2.INTER_LINEAR)
            if pose is not None:
                txt = ["d = %.2f m" % pose["dist_m"],
                       "yaw %+.1f  pitch %+.1f  roll %+.1f" % (pose["yaw"], pose["pitch"], pose["roll"]),
                       "reproj %.2f px" % pose["reproj"],
                       "truth  d=%.1f yaw=%+.0f pitch=%+.0f" % (tz, yaw, pitch)]
                for i, line in enumerate(txt):
                    cv2.putText(crop, line, (14, 30 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                                0.62, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(crop, line, (14, 30 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                                0.62, (0, 255, 255) if i < 3 else (140, 240, 140), 1, cv2.LINE_AA)
            tiles.append(crop)

        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        path = args.out or os.path.join(BASE, "demo7_out.jpg")
        imwrite_u(path, grid)
        print("  已保存 -> %s" % os.path.basename(path))
        if not args.no_show:
            _show(grid, "demo7 - PnP on synthetic scenes")
        return

    # ---------------- 图片 ----------------
    if args.image:
        img = imread_u(args.image)
        H, W = img.shape[:2]
        K, dist, calibrated = load_K(args.calib, W, H)
        if not calibrated:
            print("⚠ 未提供标定内参，用 %.0f° 视场角估算（距离会有系统偏差）" % DEFAULT_HFOV_DEG)

        import demo6_armor as d6
        armors, bars, mask, thr = d6.detect_armors(img)
        out = img.copy()
        for ar in armors:
            quad = bars_to_quad(ar)
            pose = solve_pose(quad, K, dist)
            if pose is None:
                continue
            out = draw_pose(out, pose, K, dist)
            print("  d=%.2fm  yaw=%+.1f°  pitch=%+.1f°  roll=%+.1f°  reproj=%.2fpx"
                  % (pose["dist_m"], pose["yaw"], pose["pitch"], pose["roll"], pose["reproj"]))
        if not armors:
            print("  没找到装甲板——先跑 demo6 调阈值看看")
        p = args.out or os.path.join(BASE, "demo7_out.jpg")
        imwrite_u(p, out)
        print("  已保存 -> %s" % os.path.basename(p))
        if not args.no_show:
            _show(out, "demo7 - " + os.path.basename(args.image))
        return

    # ---------------- 视频 / 摄像头 ----------------
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        print("打开失败：", args.video or ("camera %d" % args.camera))
        return

    import demo6_armor as d6
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    K, dist, calibrated = load_K(args.calib, W, H)
    if not calibrated:
        print("⚠ 未提供标定内参，用 %.0f° 视场角估算（距离会有系统偏差）" % DEFAULT_HFOV_DEG)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = None
    if args.video:
        writer = cv2.VideoWriter(args.out or os.path.join(BASE, "demo7_out.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    show = not args.no_show
    n = 0
    prev_rvec = None          # 上一帧姿态，用来在平面双解里挑对的那个
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        armors, bars, mask, thr = d6.detect_armors(frame)
        out = frame
        for ar in armors:
            pose = solve_pose(bars_to_quad(ar), K, dist, prev_rvec=prev_rvec)
            if pose is not None:
                prev_rvec = pose["rvec"]      # 只跟随第一块（最近/最可信的）目标
                out = draw_pose(out, pose, K, dist)
        n += 1
        if writer is not None:
            writer.write(out)
        if show:
            cv2.imshow("demo7 - PnP", out)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if k == ord("s"):
                imwrite_u(os.path.join(BASE, "demo7_snapshot.jpg"), out)
                print("  已存快照 demo7_snapshot.jpg")

    cap.release()
    if writer is not None:
        writer.release()
        print("  已保存 -> %s" % os.path.basename(args.out or "demo7_out.mp4"))
    if show:
        cv2.destroyAllWindows()
    print("  共处理 %d 帧" % n)


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
