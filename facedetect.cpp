#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

int main() {

    CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        cout << "Error: Cannot load cascade file!" << endl;
        return -1;
    }

    VideoCapture cap("sample1.mp4");
    if (!cap.isOpened()) {
        cout << "Error: Cannot open video!" << endl;
        return -1;
    }

    int width  = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);

    VideoWriter writer("facedetect_output.mp4",
                       VideoWriter::fourcc('m','p','4','v'),
                       fps, Size(width, height));

    int frame_count  = 0;
    int total_faces  = 0;
    double total_time = 0;
    int MAX_FRAMES   = 200;
    Mat frame, gray;

    cout << "Processing " << MAX_FRAMES << " frames..." << endl;

    while (cap.read(frame) && frame_count < MAX_FRAMES) {

        auto start = chrono::high_resolution_clock::now();

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        vector<Rect> faces;
        face_cascade.detectMultiScale(
            gray,
            faces,
            1.1,         // scale factor
            8,           // min neighbours — strict to avoid false detections
            0,
            Size(80, 80) // min face size — ignores small false positives like shirts
        );

        for (size_t i = 0; i < faces.size(); i++) {
            rectangle(frame, faces[i], Scalar(0, 255, 0), 2);
            string label = "Face " + to_string(i + 1);
            putText(frame, label,
                    Point(faces[i].x, faces[i].y - 8),
                    FONT_HERSHEY_SIMPLEX, 0.6,
                    Scalar(0, 255, 0), 2);
            total_faces++;
        }

        string info = "Faces: " + to_string(faces.size());
        putText(frame, info, Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 0.8,
                Scalar(0, 200, 255), 2);

        auto end = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(end - start).count();
        total_time += ms;
        frame_count++;

        writer.write(frame);

        if (frame_count == 30) {
            imwrite("facedetect_preview.jpg", frame);
        }

        if (frame_count % 50 == 0) {
            cout << "Processed " << frame_count << " frames..." << endl;
        }
    }

    cap.release();
    writer.release();

    double det_fps = 1000.0 / (total_time / frame_count);

    cout << "==============================" << endl;
    cout << "Frames processed   : " << frame_count << endl;
    cout << "Total faces found  : " << total_faces << endl;
    cout << "Face Detection FPS : " << det_fps << endl;
    cout << "Output video saved : facedetect_output.mp4" << endl;
    cout << "==============================" << endl;

    return 0;
}
