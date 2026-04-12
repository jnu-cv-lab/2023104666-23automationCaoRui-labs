#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // ================== 任务1：读取测试图片 ==================
    Mat img = imread("test.jpg", IMREAD_COLOR);
    if (img.empty()) {
        cout << "无法读取图片，请检查路径！" << endl;
        return -1;
    }

    // ================== 任务2：输出图像基本信息 ==================
    cout << "===  图像基本信息 ===" << endl;
    cout << "宽度 (Width): " << img.cols << " px" << endl;
    cout << "高度 (Height): " << img.rows << " px" << endl;
    cout << "通道数 (Channels): " << img.channels() << endl;

    string type_str;
    int type = img.type();
    if (type == CV_8UC1) type_str = "CV_8UC1 (8位灰度图)";
    else if (type == CV_8UC3) type_str = "CV_8UC3 (8位彩色图)";
    else type_str = "其他类型";
    cout << "数据类型 (Type): " << type_str << endl;

    // ================== 任务4：转换为灰度图 ==================
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);

    // ================== 任务5：保存灰度图 ==================
    imwrite("gray_test.jpg", gray_img);
    cout << "灰度图已保存: gray_test.jpg" << endl;

    // ================== 任务6：像素访问 + 裁剪 ==================
    // 输出(100,100)像素值
    if (img.channels() == 3) {
        Vec3b pixel = img.at<Vec3b>(100, 100);
        cout << "像素 (100,100) BGR: (" 
             << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << ")" << endl;
    } else {
        uchar pixel = gray_img.at<uchar>(100, 100);
        cout << "像素 (100,100) 灰度值: " << (int)pixel << endl;
    }

    // 裁剪 200x200 并保存
    Rect roi(0, 0, 200, 200);
    Mat crop_img = img(roi);
    imwrite("crop_test.jpg", crop_img);
    cout << "裁剪图已保存: crop_test.jpg" << endl;

    cout << "\n所有任务完成！" << endl;
    return 0;
}
