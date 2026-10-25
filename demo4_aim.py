# -*- coding: utf-8 -*-
"""
第四个 OpenCV Demo —— 从"检测"到"自瞄"（目标锁定 + 偏移量）
------------------------------------------------------------------
前面 demo3 只是把目标框出来。但 RM 自瞄真正要的是：
    框出目标 → 选中一个目标 → 算出它离画面中心偏了多少 → 把这个偏移交给云台/枪管去补

这个 demo 就干这件事，分四步（代码里都标了）：
  ① 检测：复用 demo3 的 YOLO26n（不重复造轮子）
  ② 选目标：按"置信度 + 是否靠中间"打分，选一个主目标锁定
  ③ 算偏移：dx, dy = 目标中心 - 画面中心（这就是要喂给控制环的误差量）
  ④ 画准星：十字准星固定在画面中心，画连线 + 偏移读数 + FPS

运行方式（终端里先 cd 到本文件所在目录）：
  A. 图片模式：  python demo4_aim.py --image demo3_test.jpg
  B. 视频模式：  python demo4_aim.py --video 你的视频.mp4
  C. 摄像头实时：python demo4_aim.py
  D. 只锁某一类：加 --target person   （默认锁"最可信且最居中"的任意目标）
  E. 调抖动/延迟：
       --smooth 0.5   平滑系数，越大越稳但越迟钝；0 = 硬跟（会抖）
       --lead 1.0     提前量（帧），按速度预判位置，用来抵掉平滑带来的滞后；0 = 不预判

文件名用英文，避免 Windows 命令行中文乱码。
"""

import os
import sys
import time

import cv2
import numpy as np

import demo3_detect as d3  # 复用：load_net / detect / imread_u / imwrite_u

BASE = os.path.dirname(os.path.abspath(__file__))

# 平滑系数：0 完全不平滑（抖），越接近 1 越稳但越"迟钝"。真实云台里这就是个要调的参数。
SMOOTH = 0.5
# 提前量（单位：帧）。平滑必然带来滞后，用"按当前速度预判下一帧位置"把滞后补回来。
LEAD = 1.0


def pick_target(results, w, h, target_class=None):
    """② 选目标：在候选里挑一个主目标。

    打分思路（自瞄常用启发式）：
        score = 置信度 - 0.6 × 离画面中心的归一化距离
    既别选错，也别老去瞄画面边角的东西 —— 先打离准星最近的。
    """
    cands = [r for r in results if target_class is None or r[0] == target_class]
    if not cands:
        return None

    cx, cy = w / 2.0, h / 2.0
    half_diag = (cx ** 2 + cy ** 2) ** 0.5
    best, best_score = None, -1e9
    for name, conf, (x1, y1, x2, y2) in cands:
        tx, ty = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dist = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5 / half_diag  # 0~1
        score = conf - 0.6 * dist
        if score > best_score:
            best_score, best = score, (name, conf, (x1, y1, x2, y2), (tx, ty))
    return best


def draw_crosshair(img, cx, cy, color, size=140, gap=18):
    """画面中心画一个十字准星（中间留空，不挡目标）。"""
    cv2.line(img, (cx - size, cy), (cx - gap, cy), color, 1)
    cv2.line(img, (cx + gap, cy), (cx + size, cy), color, 1)
    cv2.line(img, (cx, cy - size), (cx, cy - gap), color, 1)
    cv2.line(img, (cx, cy + gap), (cx, cy + size), color, 1)
    cv2.circle(img, (cx, cy), gap, color, 1)


def aim(net, frame, target_class=None, state=None, smooth=SMOOTH, lead=LEAD):
    """整个自瞄流程，返回 (画好的画面, 信息字典)。"""
    h, w = frame.shape[:2]
    out, results = d3.detect(net, frame)          # ① 检测

    cx, cy = w // 2, h // 2
    info = {"locked": False, "n": len(results), "dx": 0, "dy": 0, "name": "", "conf": 0.0}

    t = pick_target(results, w, h, target_class)  # ② 选目标
    if t is None:
        if state is not None:                     # 目标丢了就清速度，免得再锁上时"飞过去"
            for k in ("lx", "ly", "vx", "vy", "ax", "ay"):
                state.pop(k, None)
        draw_crosshair(out, cx, cy, (170, 170, 170))
        cv2.putText(out, "NO TARGET", (cx - 70, cy + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 170, 170), 2)
        return out, info

    name, conf, (x1, y1, x2, y2), (tx, ty) = t
    dx, dy = tx - cx, ty - cy                     # ③ 算偏移（误差量）

    # 平滑 + 提前量：
    #   平滑(EMA)会让瞄准点落后目标 —— 这就是你看到的"延误"；
    #   所以先按目标速度往前预判 lead 帧，再做平滑，把延迟抵掉。
    #   lead=0 等价于原来的纯平滑；smooth=0 等价于硬跟（不抖才怪）。
    if state is not None:
        if "lx" in state:
            vx, vy = tx - state["lx"], ty - state["ly"]      # 本帧位移 = 速度（px/帧）
            state["vx"] = 0.5 * state["vx"] + 0.5 * vx        # 速度也平滑，防止噪声带飞
            state["vy"] = 0.5 * state["vy"] + 0.5 * vy
        else:
            state["vx"], state["vy"] = 0.0, 0.0
        px, py = tx + state["vx"] * lead, ty + state["vy"] * lead   # 预判位置
        if "ax" not in state:
            state["ax"], state["ay"] = px, py
        state["ax"] = smooth * state["ax"] + (1 - smooth) * px
        state["ay"] = smooth * state["ay"] + (1 - smooth) * py
        ax, ay = state["ax"], state["ay"]
        state["lx"], state["ly"] = tx, ty
    else:
        ax, ay = tx, ty

    # ④ 画图：锁定框 + 准星 + 连线 + 瞄准点
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # 框贴顶时标签画到框内，避免和左上角读数打架
    ly = y1 - 10 if y1 > 34 else y1 + 24
    cv2.putText(out, "LOCK %s %.2f" % (name, conf), (x1 + 2, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    draw_crosshair(out, cx, cy, (0, 255, 0))
    cv2.line(out, (cx, cy), (int(ax), int(ay)), (0, 200, 255), 1)
    cv2.circle(out, (int(ax), int(ay)), 6, (0, 200, 255), -1)
    cv2.circle(out, (int(tx), int(ty)), 3, (255, 255, 255), -1)

    # 左上角读数：dx/dy 就是"该往哪边转多少"
    cv2.putText(out, "dx=%+d  dy=%+d  px" % (dx, dy), (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(out, "targets=%d" % len(results), (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    info.update({"locked": True, "dx": int(dx), "dy": int(dy), "name": name, "conf": conf})
    return out, info


def parse_args(argv):
    a = {"image": None, "video": None, "target": None, "smooth": SMOOTH, "lead": LEAD}
    i = 0
    while i < len(argv):
        k = argv[i]
        if k in ("--image", "--video", "--target") and i + 1 < len(argv):
            a[k[2:]] = argv[i + 1]
            i += 2
        elif k in ("--smooth", "--lead") and i + 1 < len(argv):
            try:
                a[k[2:]] = float(argv[i + 1])
            except ValueError:
                print("参数 %s 要填数字，已忽略" % k)
            i += 2
        else:
            i += 1
    return a


def main():
    args = parse_args(sys.argv[1:])
    os.chdir(BASE)  # 相对路径读模型，绕开 OpenCV 中文绝对路径问题
    net = d3.load_net()
    if net is None:
        return

    # ---- A. 图片模式 ----
    if args["image"]:
        img = d3.imread_u(os.path.abspath(args["image"]))
        if img is None:
            print("读不到图片：", args["image"]); return
        out, info = aim(net, img, args["target"])
        d3.imwrite_u("demo4_out.jpg", out)
        if info["locked"]:
            print("锁定 %s（置信度 %.2f），偏移 dx=%+d dy=%+d 像素 | 画面内共 %d 个目标"
                  % (info["name"], info["conf"], info["dx"], info["dy"], info["n"]))
            print("  dx>0 目标在右边，dy>0 目标在下边 —— 这两个数就是喂给云台 PID 的误差量")
        else:
            print("没锁定到目标（画面里共 %d 个检测框）" % info["n"])
        print("已保存 demo4_out.jpg")
        return

    # ---- B. 视频模式 ----
    if args["video"]:
        src = os.path.abspath(args["video"])
        if not os.path.exists(src):
            # OpenCV 也吃不了中文路径，复制一份到当前目录用英文名读
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
        writer, state, stats = None, {}, {"frames": 0, "locked": 0, "t0": time.time()}
        print("视频处理中（共 %d 帧）..." % int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out, info = aim(net, frame, args["target"], state,
                            args["smooth"], args["lead"])
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter("demo4_out.mp4",
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
            writer.write(out)
            stats["frames"] += 1
            stats["locked"] += 1 if info["locked"] else 0
        cap.release()
        if writer:
            writer.release()
        dt = time.time() - stats["t0"]
        n = max(stats["frames"], 1)
        print("处理 %d 帧，锁定率 %.0f%%，平均 %.1f FPS"
              % (stats["frames"], 100.0 * stats["locked"] / n, n / dt))
        print("已保存 demo4_out.mp4")
        return

    # ---- C. 摄像头实时 ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("打不开摄像头。可先用图片模式：python demo4_aim.py --image demo3_test.jpg")
        return
    print("摄像头已打开，实时自瞄中；按 q 退出")
    state, last, fps = {}, time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out, info = aim(net, frame, args["target"], state,
                        args["smooth"], args["lead"])
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        cv2.putText(out, "FPS %.1f" % fps, (12, out.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("YOLO26 aim demo (q to quit)", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
