# -*- coding: utf-8 -*-
"""
第二个 OpenCV Demo —— 颜色追踪小球（HSV 阈值 + 轮廓 + 外接框）
----------------------------------------------------------------
这是视觉路线「第③步」的核心例子，也是 RM 自瞄/找目标的雏形：
  1) 把画面从 BGR 转到 HSV（HSV 对光照/颜色更稳，找颜色用它）
  2) 用颜色范围做阈值，得到一张"只有目标颜色是白、其余是黑"的 mask
  3) 找轮廓，挑最大的那个，画外接框 + 标出中心坐标

运行方式（终端里先 cd 到本文件所在目录）：
  A. 图片模式：python demo2_color_track.py --image demo2_test.jpg --color green
  B. 摄像头实时：python demo2_color_track.py            （默认追踪绿色，按 q 退出）
  颜色可选 green / red / blue（--color 参数）。

文件名刻意用英文，避免 Windows 命令行中文乱码。
"""

import sys
import cv2
import numpy as np

# 颜色 HSV 范围（OpenCV 里 H 是 0-179，不是 0-360）
# red 跨越 0 度，需要两段拼起来
COLOR_RANGES = {
    "green": [(np.array([40, 60, 60]),  np.array([85, 255, 255]))],
    "red":   [(np.array([0, 80, 80]),   np.array([10, 255, 255])),
              (np.array([170, 80, 80]), np.array([180, 255, 255]))],
    "blue":  [(np.array([100, 80, 60]), np.array([130, 255, 255]))],
}


def build_mask(hsv, color):
    """把 hsv 图里属于目标颜色的像素标成 255，其余 0。"""
    ranges = COLOR_RANGES.get(color, COLOR_RANGES["green"])
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, lo, hi)
    # 开运算去小噪点，闭运算填小洞
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def process(frame, color="green"):
    """返回 (画好框的画面, mask, 文字说明)。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = build_mask(hsv, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = frame.copy()
    info = "未找到 %s 物体" % color
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > 300:  # 太小的当噪声忽略
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(out, "center=(%d,%d)" % (cx, cy), (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info = "找到 %s：中心(%d,%d)，面积 %d" % (color, cx, cy, int(area))
    return out, mask, info


def main():
    args = sys.argv[1:]
    color, image_path = "green", None
    i = 0
    while i < len(args):
        if args[i] == "--color" and i + 1 < len(args):
            color = args[i + 1]; i += 2
        elif args[i] == "--image" and i + 1 < len(args):
            image_path = args[i + 1]; i += 2
        else:
            i += 1

    # ---- 图片模式 ----
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            print("读不到图片：", image_path); return
        out, mask, info = process(img, color)
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combo = np.hstack([out, mask3])  # 左：识别框  右：颜色 mask
        cv2.imwrite("demo2_out.jpg", combo)
        print(info)
        print("已保存 demo2_out.jpg（左：识别框；右：颜色 mask）")
        return

    # ---- 摄像头实时模式 ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("打不开摄像头。可先用图片模式：")
        print("  python demo2_color_track.py --image demo2_test.jpg --color green")
        return
    print("摄像头已打开，追踪 %s 物体；按 q 退出" % color)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out, mask, info = process(frame, color)
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow("color track (q to quit)", np.hstack([out, mask3]))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
