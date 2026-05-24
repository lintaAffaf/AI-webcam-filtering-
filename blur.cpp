#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

int main() {

    VideoCapture cap("sample1.mp4"); 
    if (!cap.isOpened()) {
        cout << "Error: Cannot open video!" << endl;
        return -1;
    }

    int frame_count = 0;
    double total_time = 0;
    Mat frame, blurred;

    while (cap.read(frame)) {

        auto start = chrono::high_resolution_clock::now();

        // gaussian blur — 15x15 kernel size
        GaussianBlur(frame, blurred, Size(15, 15), 0);

        auto end = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(end - start).count();
        total_time += ms;
        frame_count++;

        if (frame_count == 30) {
            imwrite("blur_output.jpg", blurred);
        }
    }

    double fps = 1000.0 / (total_time / frame_count);
    cout << "Frames processed : " << frame_count << endl;
    cout << "CPU FPS (Blur)   : " << fps << endl;

    return 0;
}
