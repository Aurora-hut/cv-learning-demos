# -*- coding: utf-8 -*-
"""
第三个 OpenCV Demo —— 目标检测（OpenCV DNN + YOLO26n，COCO 80 类）
------------------------------------------------------------------
这是视觉路线「第④步：目标检测」，也是 RM 自瞄最核心的能力：
  不再靠"颜色阈值"找东西，而是让神经网络直接告诉你
  "画面里有什么、在哪、有多确信"（画框 + 类别 + 置信度）。

模型：YOLO26n（ONNX，约 10MB，COCO 80 类）
  person 人 / car 车 / bottle 瓶子 / sports ball 球 / dog 狗 / chair 椅子 ...
  输出格式 (1, 84, 8400)：4 个框(cx,cy,w,h 已解码到输入像素) + 80 类概率，无 objectness。

运行方式（终端里先 cd 到本文件所在目录）：
  A. 图片模式：python demo3_detect.py --image 你的图.jpg
  B. 摄像头实时：python demo3_detect.py          （按 q 退出）

文件名刻意用英文，避免 Windows 命令行中文乱码。
"""

import os
import sys
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
# 相对路径读模型：OpenCV 在 Windows 下打不开含中文的绝对路径，先 chdir 到脚本目录再读
MODEL = os.environ.get("YOLO_MODEL", os.path.join("models", "yolo26n.onnx"))
INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

# COCO 80 类
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
COLORS = np.random.RandomState(42).randint(0, 255, size=(len(CLASSES), 3))


def load_net():
    if not os.path.exists(MODEL):
        print("找不到模型：", os.path.abspath(MODEL))
        print("请确认 models/ 目录下有 yolo26n.onnx")
        return None
    return cv2.dnn.readNetFromONNX(MODEL)


def imread_u(path):
    """读图（兼容中文路径）。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite_u(path, img):
    """写图（兼容中文路径）。"""
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".jpg", img)
    if ok:
        buf.tofile(path)
    return ok


def letterbox(img, size=INPUT_SIZE, color=(114, 114, 114)):
    """等比缩放 + 灰边填充到 size×size，返回 (图, 缩放比, 左/上边距)。"""
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    dw, dh = (size - nw) // 2, (size - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, dw, dh


def detect(net, frame, conf=None):
    """返回 (画好框的画面, [(类别名, 置信度, (x1,y1,x2,y2)), ...])。

    conf 可临时覆盖置信度阈值（默认用 CONF_THRESHOLD=0.25）。
    调低能看到更多"模型看到了但没敢确认"的目标，代价是误检变多。
    """
    th = CONF_THRESHOLD if conf is None else conf
    h, w = frame.shape[:2]
    img, r, dw, dh = letterbox(frame)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True)
    net.setInput(blob)
    pred = net.forward()[0].T  # (8400, 84)

    boxes_c = pred[:, :4]        # cx, cy, w, h（输入像素，已解码）
    scores = pred[:, 4:]         # 80 类概率
    class_ids = scores.argmax(1)
    confs = scores[np.arange(len(scores)), class_ids]
    keep = confs > th
    boxes_c, class_ids, confs = boxes_c[keep], class_ids[keep], confs[keep]

    out = frame.copy()
    results = []
    if len(boxes_c) > 0:
        # cxcywh -> xyxy
        xyxy = np.empty_like(boxes_c)
        xyxy[:, 0] = boxes_c[:, 0] - boxes_c[:, 2] / 2
        xyxy[:, 1] = boxes_c[:, 1] - boxes_c[:, 3] / 2
        xyxy[:, 2] = boxes_c[:, 0] + boxes_c[:, 2] / 2
        xyxy[:, 3] = boxes_c[:, 1] + boxes_c[:, 3] / 2
        # 去 padding、除以缩放比，映射回原图
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dw) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dh) / r
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w - 1)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h - 1)

        # NMS 去重复框
        idxs = cv2.dnn.NMSBoxes(
            [[float(x), float(y), float(x2 - x), float(y2 - y)]
             for x, y, x2, y2 in xyxy],
            [float(c) for c in confs], th, NMS_THRESHOLD)
        if len(idxs) > 0:
            for i in np.array(idxs).flatten():
                x1, y1, x2, y2 = xyxy[i].astype(int)
                cid = int(class_ids[i])
                conf = float(confs[i])
                name = CLASSES[cid] if cid < len(CLASSES) else str(cid)
                color = [int(v) for v in COLORS[cid % len(COLORS)]]
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out, "%s %.2f" % (name, conf), (x1, max(y1 - 8, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                results.append((name, conf, (int(x1), int(y1), int(x2), int(y2))))
    return out, results


def main():
    global MODEL
    args = sys.argv[1:]
    image_path = None
    conf = None
    i = 0
    while i < len(args):
        if args[i] == "--image" and i + 1 < len(args):
            image_path = os.path.abspath(args[i + 1]); i += 2
        elif args[i] == "--conf" and i + 1 < len(args):
            try:
                conf = float(args[i + 1])
            except ValueError:
                print("--conf 要填数字（例如 0.1），已忽略")
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            MODEL = args[i + 1]; i += 2
        else:
            i += 1

    # 切到脚本目录，用相对路径读模型（绕开 OpenCV 中文绝对路径的兼容问题）
    os.chdir(BASE)
    net = load_net()
    if net is None:
        return
    th = CONF_THRESHOLD if conf is None else conf
    print("模型：%s ｜ 置信度阈值：%.2f" % (MODEL, th))

    if image_path:
        img = imread_u(image_path)
        if img is None:
            print("读不到图片：", image_path); return
        out, results = detect(net, img, th)
        imwrite_u("demo3_out.jpg", out)
        if results:
            print("检测到 %d 个目标：" % len(results))
            for name, c, box in results:
                print("  - %-14s 置信度 %.2f  框 %s" % (name, c, box))
        else:
            print("没检测到目标（可换张图，或 --conf 0.05 再试）")
        print("已保存 demo3_out.jpg")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("打不开摄像头。可先用图片模式：python demo3_detect.py --image 你的图.jpg")
        return
    print("摄像头已打开：按 s 存快照（cam_snapshot.jpg），按 q 退出")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out, results = detect(net, frame, th)
        cv2.putText(out, "conf>%.2f  n=%d  [s]save [q]quit" % (th, len(results)),
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("YOLO26 object detection (s save / q quit)", out)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            imwrite_u("cam_snapshot.jpg", out)
            print("已保存 cam_snapshot.jpg，本帧检出 %d 个：%s"
                  % (len(results), [(n, round(c, 2)) for n, c, _ in results]))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
