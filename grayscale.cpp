#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

int main() {

    VideoCapture cap("sample1.mp4");  // change this to your actual filename
    if (!cap.isOpened()) {
        cout << "Error: Cannot open video!" << endl;
        return -1;
    }

    int frame_count = 0;
    double total_time = 0;

    Mat frame, gray;

    while (cap.read(frame)) {

        // start timer
        auto start = chrono::high_resolution_clock::now();

        // convert to grayscale (CPU filter)
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // stop timer
        auto end = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(end - start).count();
        total_time += ms;
        frame_count++;

        // save last frame as image so we can see it
        if (frame_count == 30) {
            imwrite("grayscale_output.jpg", gray);
        }
    }

    double avg_ms = total_time / frame_count;
    double fps    = 1000.0 / avg_ms;

    cout << "Frames processed : " << frame_count << endl;
    cout << "Avg time per frame: " << avg_ms << " ms" << endl;
    cout << "CPU FPS           : " << fps << endl;

    return 0;
}
