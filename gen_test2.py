# -*- coding: utf-8 -*-
"""生成 demo2 的测试图：一个绿球 + 蓝方块 + 红圆做干扰，验证只找到绿球。"""
import cv2
import numpy as np

h, w = 480, 640
img = np.full((h, w, 3), 230, dtype=np.uint8)  # 浅灰背景

cv2.circle(img, (220, 250), 70, (0, 200, 0), -1)              # 绿球（目标）
cv2.rectangle(img, (450, 80), (590, 210), (200, 0, 0), -1)    # 蓝方块（干扰）
cv2.circle(img, (500, 380), 55, (0, 0, 220), -1)              # 红圆（干扰）
cv2.putText(img, "find the GREEN ball", (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)

cv2.imwrite("demo2_test.jpg", img)
print("已生成 demo2_test.jpg")
