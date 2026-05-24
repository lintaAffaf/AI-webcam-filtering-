#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

// this function runs on the GPU
// one thread handles exactly one pixel
__global__ void grayscaleKernel(unsigned char* input,
                                 unsigned char* output,
                                 int width, int height) {

    // which pixel does THIS thread handle?
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // make sure we don't go outside the image
    if (x < width && y < height) {

        // each pixel has 3 values: Blue, Green, Red
        int idx = (y * width + x) * 3;

        unsigned char b = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char r = input[idx + 2];

        // standard grayscale formula
        output[y * width + x] = (unsigned char)(0.299f*r + 0.587f*g + 0.114f*b);
    }
}

int main() {

    VideoCapture cap("sample1.mp4");  // change to your filename
    if (!cap.isOpened()) {
        cout << "Error: Cannot open video!" << endl;
        return -1;
    }

    int frame_count = 0;
    double total_time = 0;
    Mat frame;

    while (cap.read(frame)) {

        int width  = frame.cols;
        int height = frame.rows;
        int input_size  = width * height * 3;  // BGR = 3 channels
        int output_size = width * height;       // gray = 1 channel

        // --- step 1: allocate memory on GPU ---
        unsigned char *gpu_input, *gpu_output;
        cudaMalloc(&gpu_input,  input_size);
        cudaMalloc(&gpu_output, output_size);

        auto start = chrono::high_resolution_clock::now();

        // --- step 2: copy frame from CPU to GPU ---
        cudaMemcpy(gpu_input, frame.data, input_size, cudaMemcpyHostToDevice);

        // --- step 3: launch kernel ---
        // each block has 16x16 = 256 threads
        dim3 blockSize(16, 16);
        // enough blocks to cover entire image
        dim3 gridSize((width + 15) / 16, (height + 15) / 16);

        grayscaleKernel<<<gridSize, blockSize>>>(gpu_input, gpu_output, width, height);

        // wait for GPU to finish
        cudaDeviceSynchronize();

        // --- step 4: copy result back from GPU to CPU ---
        Mat gray(height, width, CV_8UC1);
        cudaMemcpy(gray.data, gpu_output, output_size, cudaMemcpyDeviceToHost);

        auto end = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(end - start).count();
        total_time += ms;
        frame_count++;

        // save frame 30 so we can see it
        if (frame_count == 30) {
            imwrite("grayscale_cuda_output.jpg", gray);
        }

        // --- step 5: free GPU memory ---
        cudaFree(gpu_input);
        cudaFree(gpu_output);
    }

    double avg_ms = total_time / frame_count;
    double fps    = 1000.0 / avg_ms;

    cout << "Frames processed  : " << frame_count << endl;
    cout << "Avg time per frame: " << avg_ms << " ms" << endl;
    cout << "GPU FPS (Grayscale): " << fps << endl;

    return 0;
}
