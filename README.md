# 视觉学习 Demo 集 · RM 视觉组独立路线

四个从零跑通的计算机视觉 demo，按「边缘 → 颜色 → 检测 → 自瞄」递进。
面向 RoboMaster 视觉/算法组的核心能力，也适合作为 CV 方向的作品集第 0 件。

- 作者：王思茹（中山大学 · 人工智能学院 26 级）
- 说明：代码由 AI 协作生成，本人在本机实际运行并验收；文中所有数字均为本机实测。

---

## 环境

| 项目 | 版本 / 说明 |
|---|---|
| 系统 | Windows 10 家庭中文版 25H2（Build 26200） |
| Python | 3.13.12（**注意**：本机 `python` 指向 Anaconda 3.7，跑不了；需用完整路径调 3.13.12） |
| OpenCV | 5.0.0.93 |
| numpy | 2.5.4 |
| 模型 | YOLO26n (ONNX)，10 MB，COCO 80 类 |

安装依赖：

```bash
pip install opencv-python numpy
```

模型不进仓库（体积 + 可重新下载），需要时执行：

```bash
mkdir -p models
curl -L -o models/yolo26n.onnx \
  "https://hf-mirror.com/unileon-robotics/YOLO26-ONNX/resolve/main/dynamic/yolo26n.onnx"
```

---

## 四个 demo

| # | 文件 | 做什么 | 关键点 |
|---|---|---|---|
| 1 | `demo1_edge.py` | 灰度化 + Canny 边缘检测 | 输出「原图 \| 灰度 \| 边缘」三连图 |
| 2 | `demo2_color_track.py` | HSV 颜色阈值追踪目标 | 转 HSV 抗光照、取最大轮廓、算中心坐标 |
| 3 | `demo3_detect.py` | YOLO26n 目标检测 | OpenCV DNN 跑 ONNX，不依赖 PyTorch；letterbox + NMS |
| 4 | `demo4_aim.py` | 目标锁定 + 自瞄偏移量 | 选目标打分、算 dx/dy、速度预判补偿延迟 |

运行方式（三种模式通用）：

```bash
python demo4_aim.py --image 图片.jpg     # 图片
python demo4_aim.py --video 视频.mp4     # 视频，输出 demo4_out.mp4
python demo4_aim.py                      # 摄像头实时，按 q 退出
```

Windows 上也可直接双击 `运行demo1.bat` ~ `运行demo4.bat`（内容为纯英文，避免 GBK 乱码）。

---

## demo4 的延迟问题与修法

初版用纯 EMA 平滑，准星会落后目标。在 `demo4_pan.mp4`（60 帧匀速横移）上实测：

| smooth | lead | 平均绝对滞后 | 最大 | 偏置 |
|---|---|---|---|---|
| 0.6 | 0（只平滑） | 10.17 px | 36.82 px | +10.17 |
| 0.6 | 1.0 | 3.69 px | 21.60 px | +3.58 |
| **0.5** | **1.0（默认）** | **1.41 px** | 11.63 px | **+0.20** |
| 0.4 | 0.8 | 1.29 px | 6.62 px | -0.76 |

修法：先用目标速度预判 `lead` 帧位置（`px = tx + v * lead`），再做平滑。
即"平滑负责不抖、预判负责不慢"，两个参数各自可调：

```bash
python demo4_aim.py --video demo4_pan.mp4 --smooth 0.4 --lead 0.8
```

---

## 踩过的坑（OpenCV 5.0 / Windows）

1. **OpenCV 5.0 移除了 Caffe 读取器**，`readNetFromCaffe` 不存在 → 模型只能走 ONNX。
2. **OpenCV 5.0 打不开含中文的绝对路径**的文件 → 先 `os.chdir` 到脚本目录用相对路径；图片读写用 `np.fromfile + imdecode`、`imencode + tofile`。
3. **`.bat` 里不能写中文**：cmd 按 GBK 解码 UTF-8 会把命令拆碎（报 `'?..' 不是内部或外部命令`）→ bat 一律纯英文 + `cd /d "%~dp0"`。
4. **GitHub 直连不通时**，模型可从 `hf-mirror.com` 拉取；`raw.githubusercontent.com` 与 Git LFS 均不可用。

---

## 目录

```
.
├── demo1_edge.py          边缘检测
├── demo2_color_track.py   颜色追踪
├── demo3_detect.py        YOLO26n 目标检测
├── demo4_aim.py           目标锁定 + 自瞄偏移（复用 demo3）
├── gen_test2.py           生成 demo2 测试图
├── 生成测试图.py           生成 demo1 测试图
├── 运行demo1~4.bat         一键运行（纯英文）
├── models/                模型目录（.onnx 不入库）
└── *_out.jpg / demo4_out.mp4   各 demo 实测输出
```
