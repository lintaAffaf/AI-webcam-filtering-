#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

// ===========================
// KERNEL 1 — Grayscale
// ===========================
__global__ void grayscaleKernel(unsigned char* input,
                                 unsigned char* output,
                                 int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned char b = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char r = input[idx + 2];
        output[y * width + x] = (unsigned char)(0.299f*r + 0.587f*g + 0.114f*b);
    }
}

// ===========================
// KERNEL 2 — Sepia
// ===========================
__global__ void sepiaKernel(unsigned char* input,
                             unsigned char* output,
                             int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        float b = input[idx];
        float g = input[idx + 1];
        float r = input[idx + 2];

        output[idx]     = (unsigned char)min(255.0f, 0.272f*r + 0.534f*g + 0.131f*b);
        output[idx + 1] = (unsigned char)min(255.0f, 0.349f*r + 0.686f*g + 0.168f*b);
        output[idx + 2] = (unsigned char)min(255.0f, 0.393f*r + 0.769f*g + 0.189f*b);
    }
}

// ===========================
// KERNEL 3 — Blur (3x3 average)
// ===========================
__global__ void blurKernel(unsigned char* input,
                            unsigned char* output,
                            int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int channels = 3;
        for (int c = 0; c < channels; c++) {
            float sum = 0;
            int count = 0;
            // average the 3x3 neighborhood around this pixel
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        sum += input[(ny * width + nx) * 3 + c];
                        count++;
                    }
                }
            }
            output[(y * width + x) * 3 + c] = (unsigned char)(sum / count);
        }
    }
}

// ===========================
// KERNEL 4 — Edge Detection (Sobel)
// ===========================
__global__ void edgeKernel(unsigned char* input,
                            unsigned char* output,
                            int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width-1 && y > 0 && y < height-1) {

        // Sobel X kernel
        int gx = -input[((y-1)*width + (x-1))*3]
                 + input[((y-1)*width + (x+1))*3]
                 - 2*input[(y*width + (x-1))*3]
                 + 2*input[(y*width + (x+1))*3]
                 - input[((y+1)*width + (x-1))*3]
                 + input[((y+1)*width + (x+1))*3];

        // Sobel Y kernel
        int gy = -input[((y-1)*width + (x-1))*3]
                 - 2*input[((y-1)*width + x)*3]
                 - input[((y-1)*width + (x+1))*3]
                 + input[((y+1)*width + (x-1))*3]
                 + 2*input[((y+1)*width + x)*3]
                 + input[((y+1)*width + (x+1))*3];

        output[y*width + x] = (unsigned char)min(255, (int)sqrtf(gx*gx + gy*gy));
    }
}

// ===========================
// helper: run kernel + measure time
// ===========================
void processFrames(const char* label,
                   VideoCapture& cap,
                   int kernelType) {

    int frame_count = 0;
    double total_time = 0;
    Mat frame;
    string outputFile = string(label) + "_cuda_output.jpg";

    cap.set(CAP_PROP_POS_FRAMES, 0); // rewind video to start

    while (cap.read(frame)) {
        int width  = frame.cols;
        int height = frame.rows;
        int rgb_size  = width * height * 3;
        int gray_size = width * height;

        unsigned char *gpu_input, *gpu_output;
        cudaMalloc(&gpu_input, rgb_size);

        // output size depends on filter
        if (kernelType == 0 || kernelType == 3)
            cudaMalloc(&gpu_output, gray_size);
        else
            cudaMalloc(&gpu_output, rgb_size);

        auto start = chrono::high_resolution_clock::now();

        cudaMemcpy(gpu_input, frame.data, rgb_size, cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((width + 15) / 16, (height + 15) / 16);

        if      (kernelType == 0) grayscaleKernel<<<gridSize, blockSize>>>(gpu_input, gpu_output, width, height);
        else if (kernelType == 1) sepiaKernel    <<<gridSize, blockSize>>>(gpu_input, gpu_output, width, height);
        else if (kernelType == 2) blurKernel     <<<gridSize, blockSize>>>(gpu_input, gpu_output, width, height);
        else if (kernelType == 3) edgeKernel     <<<gridSize, blockSize>>>(gpu_input, gpu_output, width, height);

        cudaDeviceSynchronize();

        // copy result back
        Mat result;
        if (kernelType == 0 || kernelType == 3) {
            result = Mat(height, width, CV_8UC1);
            cudaMemcpy(result.data, gpu_output, gray_size, cudaMemcpyDeviceToHost);
        } else {
            result = Mat(height, width, CV_8UC3);
            cudaMemcpy(result.data, gpu_output, rgb_size, cudaMemcpyDeviceToHost);
        }

        auto end = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(end - start).count();
        total_time += ms;
        frame_count++;

        if (frame_count == 30)
            imwrite(outputFile, result);

        cudaFree(gpu_input);
        cudaFree(gpu_output);
    }

    double fps = 1000.0 / (total_time / frame_count);
    cout << label << " GPU FPS: " << fps << endl;
}

int main() {

    VideoCapture cap("sample1.mp4");
    if (!cap.isOpened()) {
        cout << "Error: Cannot open video!" << endl;
        return -1;
    }

    cout << "Running all CUDA filters..." << endl;
    cout << "==============================" << endl;

    processFrames("Grayscale", cap, 0);
    processFrames("Sepia",     cap, 1);
    processFrames("Blur",      cap, 2);
    processFrames("Edge",      cap, 3);

    cout << "==============================" << endl;
    cout << "All done! Check output images." << endl;

    return 0;
}
