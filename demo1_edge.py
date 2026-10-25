# -*- coding: utf-8 -*-
"""
第一个 OpenCV Demo —— 摄像头 / 图片 → 灰度 → Canny 边缘检测
------------------------------------------------------------
这是你视觉学习路线「第①步」的最小可跑例子。它做三件事：
  1) 把彩色画面转成灰度图
  2) 用 Canny 算法提取边缘
  3) 把「原图 | 灰度 | 边缘」三张并排显示/保存，方便对比

运行方式（在终端里先 cd 到本文件所在目录）：
  A. 有图片想试：  python demo1_edge.py --image 你的图.jpg
  B. 用摄像头实时：python demo1_edge.py
     摄像头模式下按键盘 q 退出。

注意：python 指向你机器上的 Python 3.13.12（OpenCV 装在这里）。
文件名刻意用英文，避免 Windows 命令行中文乱码（之前踩过坑）。
"""

import sys
import cv2
import numpy as np


def process(frame):
    """输入一帧 BGR 图，返回『原图 | 灰度 | 边缘』并排图。"""
    # 1. 转灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 2. 先降噪（高斯模糊），再做 Canny 边缘检测
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # 把单通道图补回成 3 通道，才能和原图并排拼
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edges3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return np.hstack([frame, gray3, edges3])


def main():
    # ---- 模式 A：处理单张图片并保存 ----
    if len(sys.argv) >= 3 and sys.argv[1] == "--image":
        img = cv2.imread(sys.argv[2])
        if img is None:
            print("读不到图片，检查路径：", sys.argv[2])
            return
        out = process(img)
        out_path = "demo1_out.jpg"
        cv2.imwrite(out_path, out)
        print("处理完成，已保存：", out_path, "（用看图软件打开看三连图）")
        return

    # ---- 模式 B：摄像头实时 ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("打不开 0 号摄像头。两种办法：")
        print("  1) 先用图片模式试试：python demo1_edge.py --image demo1_test.jpg")
        print("  2) 没有摄像头也能学，几十块买个 USB 摄像头即可（详见计划文档『硬件』说明）")
        return
    print("摄像头已打开，按 q 退出")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("原图 | 灰度 | 边缘  (按 q 退出)", process(frame))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
