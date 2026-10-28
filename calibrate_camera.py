# -*- coding: utf-8 -*-
"""
相机标定（张正友标定法）—— 给 demo7 的 PnP 提供真实内参
------------------------------------------------------------------
为什么必须标定：
    PnP 解的是  s·[u,v,1]ᵀ = K·(R·P + t)。
    这里的 K（内参矩阵）如果只是"猜"的，那么解出来的距离会**整体乘上一个错误的系数**——
    形状对、姿态大致对，但"3 米"可能实际是"3.6 米"。
    打靶要的是绝对距离，所以这一步绕不过去。

    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    fx/fy：焦距（单位是像素），cx/cy：光心（主点）。

怎么用：

  第 1 步  生成棋盘格
      python calibrate_camera.py --make-board
      → 得到 chessboard_9x6_25mm.png，**按 100% 比例打印**后贴在硬纸板上（一定要平）。
        没有打印机的话，用手机全屏显示也行，但要把屏幕亮度调高、避开反光。

  第 2 步  拍 15~25 张
      python calibrate_camera.py
      举着棋盘格在镜头前，覆盖这些变化：
        · 远一点 / 近一点         · 画面左上 / 右上 / 左下 / 右下 / 正中
        · 左右各倾斜 20~40°       · 上下各倾斜 20~30°（**俯仰很关键**，别只左右转）
      每摆好一个姿势按 **空格** 拍一张，按 q 结束。
      拍到 15 张以上会自动尝试标定。

  第 3 步  用标定结果跑 demo7
      python demo7_pnp.py --calib camera.json

  离线验证标定流程本身对不对：
      python calibrate_camera.py --selftest
"""

import argparse
import glob
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

# 棋盘格参数（必须和实际打印出来的一致！）
BOARD_COLS = 9          # 内角点列数
BOARD_ROWS = 6          # 内角点行数
SQUARE_MM = 25.0        # 单个格子边长


def imread_u(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(path)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("decode failed: " + path)
    return img


def imwrite_u(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


# ----------------------------------------------------------------------------
# 生成可打印的棋盘格
# ----------------------------------------------------------------------------
def make_board(path=None, dpi=300):
    sq = int(round(SQUARE_MM / 25.4 * dpi))
    # 关键：内角点有 cols x rows 个，方格就必须是 (cols+1) x (rows+1) 个。
    # 画成 cols x rows 个方格的话，内角点只有 (cols-1) x (rows-1)，findChessboardCorners 永远找不到。
    w, h = (BOARD_COLS + 1) * sq, (BOARD_ROWS + 1) * sq
    board = np.full((h, w), 255, np.uint8)
    for r in range(BOARD_ROWS + 1):
        for c in range(BOARD_COLS + 1):
            if (r + c) % 2 == 0:
                board[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 0
    margin = int(round(12 / 25.4 * dpi))
    board = cv2.copyMakeBorder(board, margin, margin, margin, margin,
                               cv2.BORDER_CONSTANT, value=255)
    cv2.putText(board, "%dx%d inner corners  square=%.0fmm" % (BOARD_COLS, BOARD_ROWS, SQUARE_MM),
                (margin, margin + h + int(margin * 0.62)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1, cv2.LINE_AA)
    path = path or os.path.join(BASE, "chessboard_9x6_25mm.png")
    ok, buf = cv2.imencode(".png", board)
    if ok:
        buf.tofile(path)
    print("已生成 -> %s" % os.path.basename(path))
    print("  内角点 %d x %d，格子 %.0f mm，%d dpi（约 %.0f x %.0f mm）"
          % (BOARD_COLS, BOARD_ROWS, SQUARE_MM, dpi,
             ((BOARD_COLS + 1) * SQUARE_MM + 24), ((BOARD_ROWS + 1) * SQUARE_MM + 34)))
    print("  打印时务必选「实际大小 / 100%」，不要选「适应页面」")
    return path


def object_points():
    """棋盘格内角点的 3D 坐标（Z=0 平面），单位 mm。

    必须是 float32：OpenCV 5 的 calibrateCamera 明确要求 Point3f，
    给 float64 会直接抛 "objectPoints should contain vector of vectors of points of type Point3f"。
    """
    op = np.zeros((BOARD_COLS * BOARD_ROWS, 3), np.float32)
    op[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    return (op * SQUARE_MM).astype(np.float32)


def find_corners(gray):
    """找棋盘格内角点。OpenCV 5 里 findChessboardCorners 的返回和 4.x 一样。"""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, (BOARD_COLS, BOARD_ROWS), flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    return corners


# ----------------------------------------------------------------------------
# 标定主流程
# ----------------------------------------------------------------------------
def calibrate(images, W, H, quiet=False):
    """images：BGR 图列表。返回 (rms, K, dist, 有效张数)。"""
    op = object_points()
    obj_pts, img_pts = [], []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c = find_corners(gray)
        if c is None:
            continue
        obj_pts.append(op)
        img_pts.append(c)

    if len(obj_pts) < 6:
        if not quiet:
            print("  有效张数只有 %d，至少需要 6 张（建议 15 张以上）" % len(obj_pts))
        return None

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, (W, H), None, None)
    if not quiet:
        print("  参与标定：%d 张，重投影 RMS = %.3f px" % (len(obj_pts), rms))
    return rms, K, dist, len(obj_pts)


def save_calib(path, K, dist, W, H, rms, n):
    d = {
        "K": K.tolist(),
        "dist": dist.ravel().tolist(),
        "width": int(W),
        "height": int(H),
        "rms_reproj_px": float(rms),
        "n_images": int(n),
        "board": {"cols": BOARD_COLS, "rows": BOARD_ROWS, "square_mm": SQUARE_MM},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    return d


# ----------------------------------------------------------------------------
# 离线自测：用已知内参渲染棋盘格 → 再标定回去，验证流程正确
# ----------------------------------------------------------------------------
def run_selftest(n_view=18):
    print("=" * 62)
    print("标定流程自测（两段）")
    print("=" * 62)

    W, H = 1280, 720

    # ---------- 第一段：渲染一张正面棋盘格，确认 cornerSubPix 能吃到角点 ----------
    print("① 角点检测：渲染一张正面棋盘格，看能否找到 %dx%d 个内角点"
          % (BOARD_COLS, BOARD_ROWS))
    sq = 65
    bw, bh = (BOARD_COLS + 1) * sq, (BOARD_ROWS + 1) * sq
    tile = np.full((bh, bw), 255, np.uint8)
    for r in range(BOARD_ROWS + 1):
        for c in range(BOARD_COLS + 1):
            if (r + c) % 2 == 0:
                tile[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 0
    canvas = np.full((H, W), 210, np.uint8)
    x0, y0 = (W - bw) // 2, (H - bh) // 2
    canvas[y0:y0 + bh, x0:x0 + bw] = tile
    found = find_corners(canvas)
    if found is None:
        print("   ✗ 找不到角点，检查 BOARD_COLS / BOARD_ROWS 是否与图案一致")
        return False
    print("   ✓ 找到 %d 个角点（期望 %d）" % (len(found), BOARD_COLS * BOARD_ROWS))

    # ---------- 第二段：用已知内参生成角点，再标定回来 ----------
    print()
    print("② 标定数学：用已知内参投影角点（带 0.2 px 检测噪声），再标定回来")
    fx_true, fy_true = 1000.0, 1005.0
    cx_true, cy_true = W / 2 + 6.0, H / 2 - 4.0     # 故意偏一点，看能不能标回来
    K_true = np.array([[fx_true, 0, cx_true], [0, fy_true, cy_true], [0, 0, 1]], np.float64)
    dist_true = np.array([-0.12, 0.06, 0.0008, -0.0005, 0.0], np.float64)

    op = object_points()
    rng = np.random.default_rng(7)
    obj_pts, img_pts = [], []
    for i in range(n_view):
        # 位姿覆盖要够散：距离、左右倾斜、上下俯仰都得有，否则 fy 和 cy 会耦合解不开
        tz = float(rng.uniform(280, 560))
        rvec = np.array([rng.uniform(-0.50, 0.50),      # 绕 X（俯仰）
                         rng.uniform(-0.60, 0.60),      # 绕 Y（左右）
                         rng.uniform(-0.25, 0.25)], np.float64)
        tvec = np.array([rng.uniform(-70, 70), rng.uniform(-50, 50), tz], np.float64)
        corners, _ = cv2.projectPoints(op, rvec, tvec, K_true, dist_true)
        corners = corners.reshape(-1, 2)
        if corners[:, 0].min() < 0 or corners[:, 0].max() > W:
            continue
        if corners[:, 1].min() < 0 or corners[:, 1].max() > H:
            continue
        obj_pts.append(op)
        img_pts.append((corners + rng.normal(0, 0.2, corners.shape)).astype(np.float32))

    if len(obj_pts) < 8:
        print("   ✗ 有效视角太少（%d）" % len(obj_pts))
        return False

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (W, H), None, None)
    efx = abs(K[0, 0] - fx_true) / fx_true * 100
    efy = abs(K[1, 1] - fy_true) / fy_true * 100
    ecx = abs(K[0, 2] - cx_true)
    ecy = abs(K[1, 2] - cy_true)

    print("   %d 个视角，重投影 RMS = %.3f px" % (len(obj_pts), rms))
    print()
    print("   %-6s %-12s %-12s %s" % ("参数", "真值", "标定值", "误差"))
    print("   %-6s %-12.2f %-12.2f %.2f%%" % ("fx", fx_true, K[0, 0], efx))
    print("   %-6s %-12.2f %-12.2f %.2f%%" % ("fy", fy_true, K[1, 1], efy))
    print("   %-6s %-12.2f %-12.2f %.2f px" % ("cx", cx_true, K[0, 2], ecx))
    print("   %-6s %-12.2f %-12.2f %.2f px" % ("cy", cy_true, K[1, 2], ecy))
    # 判定标准按工程可用性定：焦距 0.5%（影响距离缩放）、光心 3 px（影响偏移），
    # 再严就没意义了——真实拍摄时纸张不平、镜头畸变本就会带来这个量级的残差。
    ok = efx < 0.5 and efy < 0.5 and ecx < 3.0 and ecy < 3.0
    print()
    print("   结论：%s" % ("✓ 流程正确（焦距误差 < 0.5%，光心误差 < 3 px）" if ok
                        else "✗ 偏差偏大，检查 object_points 的边长单位是否为 mm"))
    print("=" * 62)
    return ok


# ----------------------------------------------------------------------------
# 交互拍摄
# ----------------------------------------------------------------------------
def run_camera(camera=0, out="camera.json", need=15, pattern=None):
    if pattern:
        files = sorted(glob.glob(os.path.join(pattern, "*.jpg")) +
                       glob.glob(os.path.join(pattern, "*.png")))
        if not files:
            print("没找到图片：", pattern)
            return
        imgs = [imread_u(f) for f in files]
        H, W = imgs[0].shape[:2]
        print("从文件夹读入 %d 张图（%dx%d）" % (len(imgs), W, H))
        out_res = calibrate(imgs, W, H)
        if out_res:
            rms, K, dist, n = out_res
            save_calib(os.path.join(BASE, out), K, dist, W, H, rms, n)
            print("  已保存 -> %s" % out)
            print("  用法：python demo7_pnp.py --calib %s" % out)
        return

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("摄像头打不开")
        return

    print("棋盘格对准镜头，摆好姿势后按【空格】拍照，按 q 结束。")
    print("目标：至少 %d 张，覆盖画面四角 + 左右倾斜 + 上下俯仰。" % need)
    shots = []
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    last_ok = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_corners(gray)
        last_ok = corners is not None
        vis = frame.copy()
        if last_ok:
            cv2.drawChessboardCorners(vis, (BOARD_COLS, BOARD_ROWS), corners, True)
        color = (0, 255, 0) if last_ok else (0, 0, 255)
        cv2.putText(vis, "found=%s  shots=%d" % ("YES" if last_ok else "no", len(shots)),
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.imshow("calibrate - SPACE to shoot, q to finish", vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" ") and last_ok:
            shots.append(frame.copy())
            print("  已拍 %d 张" % len(shots))
        elif k == ord(" ") and not last_ok:
            print("  这一帧没找到棋盘格，换个角度")

    cap.release()
    cv2.destroyAllWindows()

    if len(shots) < 6:
        print("只拍了 %d 张，不够标定（至少 6 张，建议 15 张以上）" % len(shots))
        return
    out_res = calibrate(shots, W, H)
    if not out_res:
        return
    rms, K, dist, n = out_res
    save_calib(os.path.join(BASE, out), K, dist, W, H, rms, n)
    print("  已保存 -> %s" % out)
    print("  fx=%.1f fy=%.1f cx=%.1f cy=%.1f" % (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    print("  用法：python demo7_pnp.py --calib %s" % out)
    if rms > 1.0:
        print("  ⚠ RMS 偏大（%.2f px），一般是棋盘格没贴平，或者俯仰角度覆盖不够" % rms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-board", action="store_true", help="生成可打印棋盘格")
    ap.add_argument("--selftest", action="store_true", help="离线验证标定流程")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default="camera.json")
    ap.add_argument("--images", default=None, help="用文件夹里的图片标定，不开摄像头")
    ap.add_argument("--need", type=int, default=15)
    args = ap.parse_args()

    if args.make_board:
        make_board()
        return
    if args.selftest:
        run_selftest()
        return
    run_camera(args.camera, args.out, args.need, args.images)


if __name__ == "__main__":
    main()
