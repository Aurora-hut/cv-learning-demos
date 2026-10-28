// 第一个 C++ OpenCV 程序 —— 和 Python 版 demo1 做同一件事
// 目的：① 打通 C++ 工具链 ② 亲眼看到 C++ 比 Python 快多少（RM 视觉要用 C++ 的原因）
//
// 编译/运行：双击 build.bat（会自动调 g++），或看 README_CPP.md 里的命令

#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    // 图片路径：默认用同目录的 demo1_test.jpg（相对路径，避开中文绝对路径的坑）
    std::string path = (argc > 1) ? argv[1] : "demo1_test.jpg";

    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        std::cerr << "读不到图片: " << path << std::endl;
        std::cerr << "请把 demo1_test.jpg 放在本目录，或用参数指定路径" << std::endl;
        return 1;
    }
    std::cout << "OpenCV 版本: " << CV_VERSION << std::endl;
    std::cout << "图像尺寸: " << img.cols << " x " << img.rows
              << "  通道: " << img.channels() << std::endl;

    cv::Mat gray, edges;

    // 先跑一次预热（第一次调用要初始化内部结构，不算进平均值）
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 50, 150);

    // 循环计时：灰度 + 边缘，跑 200 次取平均
    const int N = 200;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, edges, 50, 150);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;

    std::cout.precision(3);
    std::cout << "灰度+Canny: " << ms << " ms/帧   ≈ " << (1000.0 / ms) << " FPS" << std::endl;

    // —— 第二个基准：手写逐像素循环 ——
    // 这里才是 C++ 真正拉开差距的地方：Python 里写 for 循环遍历 23 万个像素极慢，
    // C++ 编译后是简单的指针扫描。RM 视觉里大量自定义处理（二值化、灯条扫描）都是这种活。
    const int N2 = 50;
    long long bright = 0;
    int total = gray.rows * gray.cols;
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N2; ++i) {
        bright = 0;
        const unsigned char* p = gray.data;   // 直接拿像素指针，无边界检查
        for (int k = 0; k < total; ++k) {
            if (p[k] > 128) ++bright;         // 统计亮像素个数
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms2 = std::chrono::duration<double, std::milli>(t3 - t2).count() / N2;
    std::cout << "手写逐像素统计: " << ms2 << " ms/帧  (亮像素 " << bright << " 个)" << std::endl;

    // 三连图：原图 | 灰度 | 边缘（和 Python 版一样的输出形式）
    cv::Mat gray3, edge3, out;
    cv::cvtColor(gray, gray3, cv::COLOR_GRAY2BGR);
    cv::cvtColor(edges, edge3, cv::COLOR_GRAY2BGR);
    cv::hconcat(std::vector<cv::Mat>{img, gray3, edge3}, out);

    if (!cv::imwrite("cpp_out.jpg", out)) {
        std::cerr << "写文件失败" << std::endl;
        return 1;
    }
    std::cout << "已保存 cpp_out.jpg（原图 | 灰度 | 边缘）" << std::endl;
    return 0;
}
