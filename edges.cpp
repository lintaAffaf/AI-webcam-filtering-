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
    Mat frame, gray, edges;

    while (cap.read(frame)) {

        auto start = chrono::high_resolution_clock::now();

        // convert to gray first, then detect edges
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        Canny(gray, edges, 100, 200);  // 100 and 200 are thresholds

        auto end = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(end - start).count();
        total_time += ms;
        frame_count++;

        if (frame_count == 30) {
            imwrite("edges_output.jpg", edges);
        }
    }

    double fps = 1000.0 / (total_time / frame_count);
    cout << "Frames processed  : " << frame_count << endl;
    cout << "CPU FPS (Edges)   : " << fps << endl;

    return 0;
}
