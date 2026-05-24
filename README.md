# AI Webcam Filters — CUDA + OpenCV

A project demonstrating real-time video filter processing using CPU (OpenCV)
and GPU (CUDA) with face detection.

## Filters Implemented
- Grayscale
- Sepia
- Edge Detection (Canny / Sobel)
- Blur (Gaussian / CUDA 3x3)

## Tech Stack
- C++
- CUDA (nvcc)
- OpenCV 4

## CPU vs GPU FPS Results
| Filter    | CPU FPS | GPU FPS | Speedup |
|-----------|---------|---------|---------|
| Grayscale | 3543    | 1402    | 0.4x    |
| Sepia     | 33      | 1194    | 35x     |
| Blur      | 561     | 1129    | 2x      |
| Edge      | 170     | 1538    | 9x      |

## How to Run on Google Colab
1. Open a new Colab notebook
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run the master setup cell
4. Run the compile cell
5. Run any filter binary with !./filtername

## Face Detection
Uses OpenCV Haar Cascade (haarcascade_frontalface_default.xml)
to detect faces in video frames and draw bounding boxes.

## Project Structure
- grayscale.cpp — CPU grayscale filter
- sepia.cpp — CPU sepia filter
- edges.cpp — CPU edge detection
- blur.cpp — CPU blur filter
- grayscale_cuda.cu — CUDA grayscale kernel
- filters_cuda.cu — All 4 CUDA kernels in one file
- facedetect.cpp — Face detection with Haar Cascade
