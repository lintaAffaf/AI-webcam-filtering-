#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

int main() {

    VideoCapture cap("sample1.mp4");  // change to your filename
    if (!cap.isOpened()) {
        cout << "Error: Cannot open video!" << endl;
        return -1;
    }

    int frame_count = 0;
    double total_time = 0;
    Mat frame, sepia;

    while (cap.read(frame)) {

        auto start = chrono::high_resolution_clock::now();

        // sepia filter — apply color matrix to each pixel
        sepia = Mat(frame.size(), frame.type());
        for (int i = 0; i < frame.rows; i++) {
            for (int j = 0; j < frame.cols; j++) {
                Vec3b pixel = frame.at<Vec3b>(i, j);
                float b = pixel[0], g = pixel[1], r = pixel[2];

                sepia.at<Vec3b>(i, j)[0] = min(255.0f, 0.272f*r + 0.534f*g + 0.131f*b); // blue
                sepia.at<Vec3b>(i, j)[1] = min(255.0f, 0.349f*r + 0.686f*g + 0.168f*b); // green
                sepia.at<Vec3b>(i, j)[2] = min(255.0f, 0.393f*r + 0.769f*g + 0.189f*b); // red
            }
        }

        auto end = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(end - start).count();
        total_time += ms;
        frame_count++;

        if (frame_count == 30) {
            imwrite("sepia_output.jpg", sepia);
        }
    }

    double fps = 1000.0 / (total_time / frame_count);
    cout << "Frames processed : " << frame_count << endl;
    cout << "CPU FPS (Sepia)  : " << fps << endl;

    return 0;
}
