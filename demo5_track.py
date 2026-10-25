# -*- coding: utf-8 -*-
"""
第五个 Demo —— 检测 + 跟踪分离（把自瞄的帧率提上去）
------------------------------------------------------------------
问题：demo3/demo4 每帧都跑一遍神经网络，CPU 上只有几 FPS。
      目标一动，画面就"卡一下再跟上"，看起来就是不准。
思路（RM 队伍的标准做法）：
      不要每帧都检测。**隔 N 帧检测一次**做"校正"，中间用**光流跟踪**（Lucas-Kanade）
      跟着目标走。检测 ~几 Hz + 跟踪 ~几十 Hz，既准又跟手。

具体流程：
  ① 该检测的帧：YOLO 检测 → 选目标 → 在框内找特征点（红框 = 刚被检测校正）
  ② 中间帧：把这批特征点光流推到下一帧 → 取位移中位数 → 平移框（绿框 = 纯跟踪）
  ③ 有效点太少 / 框跑出画面 → 下一帧强制重新检测（防止跟丢后一直错下去）

运行：
  A. 视频：  python demo5_track.py --video demo4_pan.mp4
  B. 摄像头：python demo5_track.py
  C. 换检测间隔：--every 5（更准更慢） / --every 20（更快更容易飘）
  D. 只锁某一类：--target person
"""

import os
import sys
import time

import cv2
import numpy as np

import demo3_detect as d3
from demo4_aim import pick_target, draw_crosshair

BASE = os.path.dirname(os.path.abspath(__file__))

DETECT_EVERY = 10   # 每 N 帧检测一次，中间靠光流
MIN_PTS = 8         # 光流有效点少于这个数就重新检测


def init_points(gray, box):
    """在目标框里挑一批好跟的角点。"""
    x, y, w, h = [int(v) for v in box]
    x, y = max(x, 0), max(y, 0)
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return None
    pts = cv2.goodFeaturesToTrack(roi, maxCorners=40, qualityLevel=0.01,
                                  minDistance=5, blockSize=3)
    if pts is None:
        return None
    pts[:, :, 0] += x          # ROI 坐标 -> 全图坐标
    pts[:, :, 1] += y
    return pts


def track_points(prev_gray, gray, pts):
    """光流把特征点推到当前帧，返回 (新点, 位移中位数 dx, dy)。"""
    if pts is None or len(pts) == 0:
        return None, 0.0, 0.0
    new_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
    if new_pts is None:
        return None, 0.0, 0.0
    ok = st.reshape(-1) == 1
    if ok.sum() < MIN_PTS:
        return None, 0.0, 0.0
    old, new = pts.reshape(-1, 2)[ok], new_pts.reshape(-1, 2)[ok]
    d = new - old
    return new.reshape(-1, 1, 2), float(np.median(d[:, 0])), float(np.median(d[:, 1]))


def run(img, net, gray_prev, target, fps_text, every, target_class):
    """处理一帧。target 是跨帧状态字典。"""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = img.copy()
    cx, cy = w // 2, h // 2

    target["since"] += 1
    need_detect = (target.get("box") is None) or (target["since"] >= every) or target.get("lost")

    mode, conf, name = "LOST", 0.0, ""
    if need_detect:
        # ---- ① 检测帧：跑 YOLO，顺便重新采特征点 ----
        # 注意用 d3.detect 返回的 results，但画面用干净的 img（只画锁定目标，不画全部检测框）
        _, results = d3.detect(net, img)
        t = pick_target(results, w, h, target_class)
        target["since"] = 0
        if t is None:
            target["box"], target["pts"], target["lost"] = None, None, True
        else:
            name, conf, (x1, y1, x2, y2), _ = t
            box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            target["box"] = box
            target["pts"] = init_points(gray, box)
            target["lost"] = False
            mode = "DETECT"
    else:
        # ---- ② 跟踪帧：光流推移目标框 ----
        new_pts, dx, dy = track_points(gray_prev, gray, target.get("pts"))
        box = target.get("box")
        if new_pts is None or box is None:
            target["lost"] = True
            target["box"] = None
        else:
            target["pts"] = new_pts
            target["box"] = [box[0] + dx, box[1] + dy, box[2], box[3]]
            b = target["box"]
            if b[0] < -b[2] * 0.5 or b[1] < -b[3] * 0.5 or b[0] > w or b[1] > h:
                target["lost"] = True
            mode = "TRACK"

    # ---- ③ 输出偏移与画面 ----
    box = target.get("box")
    if box is not None and not target.get("lost"):
        x, y, bw, bh = [int(round(v)) for v in box]
        tx, ty = x + bw // 2, y + bh // 2
        dx, dy = tx - cx, ty - cy
        color = (0, 0, 255) if mode == "DETECT" else (0, 255, 0)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
        draw_crosshair(out, cx, cy, (255, 255, 255))
        cv2.line(out, (cx, cy), (tx, ty), (0, 200, 255), 1)
        cv2.circle(out, (tx, ty), 5, (0, 200, 255), -1)
        cv2.putText(out, "dx=%+d dy=%+d  %s" % (dx, dy, mode), (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    else:
        draw_crosshair(out, cx, cy, (150, 150, 150))
        cv2.putText(out, "NO TARGET", (cx - 72, cy + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

    cv2.putText(out, fps_text, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out, gray


def parse_args(argv):
    a = {"video": None, "target": None, "every": DETECT_EVERY}
    i = 0
    while i < len(argv):
        k = argv[i]
        if k in ("--video", "--target") and i + 1 < len(argv):
            a[k[2:]] = argv[i + 1]; i += 2
        elif k == "--every" and i + 1 < len(argv):
            try:
                a["every"] = max(1, int(argv[i + 1]))
            except ValueError:
                pass
            i += 2
        else:
            i += 1
    return a


def main():
    args = parse_args(sys.argv[1:])
    os.chdir(BASE)
    net = d3.load_net()
    if net is None:
        return

    # ---- 视频模式：统计 纯检测 vs 检测+跟踪 的耗时 ----
    if args["video"]:
        src = os.path.abspath(args["video"])
        if not os.path.exists(src):
            data = np.fromfile(src, dtype=np.uint8)
            if data.size == 0:
                print("找不到视频：", src); return
            tmp = "_tmp_in" + os.path.splitext(src)[1]
            open(tmp, "wb").write(data.tobytes())
            src = tmp
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print("打不开视频：", args["video"]); return
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
        writer = None
        target = {"since": 0, "box": None, "pts": None, "lost": True}
        gray_prev = None
        n, t_detect, t_all = 0, 0.0, 0.0
        det_frames = 0
        print("处理中（检测间隔 %d 帧）..." % args["every"])
        while True:
            ok, img = cap.read()
            if not ok:
                break
            t0 = time.time()
            out, gray = run(img, net, gray_prev, target, "", args["every"], args["target"])
            t_all += time.time() - t0
            if target["since"] == 0:
                det_frames += 1
            if writer is None:
                hh, ww = out.shape[:2]
                writer = cv2.VideoWriter("demo5_out.mp4",
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (ww, hh))
            writer.write(out)
            gray_prev = gray
            n += 1
        cap.release()
        if writer:
            writer.release()
        if n == 0:
            print("没读到帧"); return
        fps = n / max(t_all, 1e-6)
        print("共 %d 帧，其中检测帧 %d 帧，纯跟踪帧 %d 帧" % (n, det_frames, n - det_frames))
        print("平均 %.1f FPS（每帧 %.1f ms）" % (fps, 1000.0 * t_all / n))
        print("已保存 demo5_out.mp4")
        return

    # ---- 摄像头模式 ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("打不开摄像头。可先用：python demo5_track.py --video demo4_pan.mp4")
        return
    print("摄像头已打开；绿框=跟踪中，红框=刚检测校正；按 q 退出")
    target = {"since": 0, "box": None, "pts": None, "lost": True}
    gray_prev, last, fps = None, time.time(), 0.0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        out, gray = run(img, net, gray_prev, target, "FPS %.1f" % fps, args["every"], args["target"])
        gray_prev = gray
        cv2.imshow("detect + optical-flow tracking (q to quit)", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
