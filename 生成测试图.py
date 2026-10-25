# -*- coding: utf-8 -*-
"""生成一张带几何图形的测试图，供 demo1 的 --image 模式使用。"""
import cv2
import numpy as np

h, w = 360, 640
img = np.ones((h, w, 3), dtype=np.uint8) * 255  # 白底

# 画几个形状，边缘明显
cv2.rectangle(img, (60, 60), (220, 220), (0, 0, 200), 4)      # 红框
cv2.circle(img, (450, 140), 80, (0, 180, 0), 4)               # 绿圆
cv2.line(img, (60, 300), (580, 300), (200, 0, 0), 4)          # 蓝线
cv2.putText(img, "OpenCV Demo Test", (120, 330),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 3)

cv2.imwrite("demo1_test.jpg", img)
print("已生成 demo1_test.jpg")
