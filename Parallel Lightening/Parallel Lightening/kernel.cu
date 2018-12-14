#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

typedef struct {
	unsigned char red, green, blue;
} PPMPixel;

typedef struct {
	int x, y;
	PPMPixel *data;
} PPMImage;

#define CREATOR "NIKITA TISHAKOV"
#define RGB_COMPONENT_COLOR 255
#define NORMAL_LIGHTING 100
#define HIGH 50

__host__ void errorHandler(const cudaError& error, const string& msg) {
	if (error != cudaSuccess) {
		cerr << msg << endl;
		exit(1);
	}
}

static PPMImage* readPPM(const char *filename){
	char buff[16];
	PPMImage *img;
	FILE *fp;
	int c, rgb_comp_color;

	//open PPM file for reading
	fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	//read image format
	if (!fgets(buff, sizeof(buff), fp)) {
		perror(filename);
		exit(1);
	}

	//check the image format
	if (buff[0] != 'P' || buff[1] != '6') {
		fprintf(stderr, "Invalid image format (must be 'P6')\n");
		exit(1);
	}

	//alloc memory form image
	img = (PPMImage *)malloc(sizeof(PPMImage));
	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	//check for comments
	c = getc(fp);
	while (c == '#') {
		while (getc(fp) != '\n');
		c = getc(fp);
	}

	ungetc(c, fp);
	//read image size information
	if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
		fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
		exit(1);
	}

	//read rgb component
	if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
		fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
		exit(1);
	}

	//check rgb component depth
	if (rgb_comp_color != RGB_COMPONENT_COLOR) {
		fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
		exit(1);
	}

	while (fgetc(fp) != '\n');
	//memory allocation for pixel data
	img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	//read pixel data from file
	if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
		fprintf(stderr, "Error loading image '%s'\n", filename);
		exit(1);
	}

	fclose(fp);
	return img;
}

void writePPM(const char *filename, PPMImage *img){
	FILE *fp;
	//open file for output
	fp = fopen(filename, "wb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	//write the header file
	//image format
	fprintf(fp, "P6\n");

	//comments
	fprintf(fp, "# Created by %s\n", CREATOR);

	//image size
	fprintf(fp, "%d %d\n", img->x, img->y);

	// rgb component depth
	fprintf(fp, "%d\n", RGB_COMPONENT_COLOR);

	// pixel data
	fwrite(img->data, 3 * img->x, img->y, fp);
	fclose(fp);
}

__host__ void changeColorPPM(PPMImage *img){
	if (img) {
		for (int i = 0; i < img->x * img->y; i++) {
			if (img->data[i].red + img->data[i].green + img->data[i].blue < NORMAL_LIGHTING) {
				img->data[i].red = img->data[i].red + HIGH;
				img->data[i].green = img->data[i].green + HIGH;
				img->data[i].blue = img->data[i].blue + HIGH;
			}
		}
	}
}

__global__ void callKernel(PPMPixel *image, const int xSize, const int ySize) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < xSize*ySize) {
		if (image[idx].red + image[idx].green + image[idx].blue < NORMAL_LIGHTING) {
			image[idx].red += HIGH;
			image[idx].green += HIGH;
			image[idx].blue += HIGH;
		}
	}
}

__host__ bool validate_filtering(const PPMImage *imgForCPU, const PPMImage *imgForGPU) {
	for (int i = 0; i < imgForCPU->x * imgForCPU->y; i++) {
		int clr1 = imgForCPU->data[i].red + imgForCPU->data[i].green + imgForCPU->data[i].blue;
		int clr2 = imgForGPU->data[i].red + imgForGPU->data[i].green + imgForGPU->data[i].blue;
		if (clr1 != clr2) return false;
	}
	return true;
}


int main() {
	PPMImage *image, *image_copy;
	image = readPPM("image.ppm");
	image_copy = readPPM("image.ppm");
	
	const size_t imageSize = image->x * image->y * sizeof(PPMPixel);
	PPMPixel *cuImage = nullptr;

	errorHandler(cudaMalloc(&cuImage, imageSize), "CUDAMALLOC: cuImage");
	errorHandler(cudaMemcpy(cuImage, image->data, imageSize, cudaMemcpyHostToDevice), "CUDAMEMCPY: TO DEVICE");

	//set blocks and threads
	int threadsPerBlock = 256;
	int blocksPerGrid = (image->x*image->y + threadsPerBlock - 1) / threadsPerBlock;
	dim3 blockSize = dim3(threadsPerBlock, threadsPerBlock, 1);
	dim3 gridSize = dim3(blocksPerGrid, blocksPerGrid, 1);

	//image lightening on the CPU
	changeColorPPM(image);

	//image lightening on the GPU
	callKernel << <blocksPerGrid, threadsPerBlock >> > (cuImage, image->x, image->y);
	errorHandler(cudaGetLastError(), "LAST ERROR AFTER 'callKernel'");
	errorHandler(cudaDeviceSynchronize(), "DEVICE SYNCHRONIZE");
	errorHandler(cudaMemcpy(image_copy->data, cuImage, imageSize, cudaMemcpyDeviceToHost), "CUDAMEMCPY: TO DEVICE");

	if (validate_filtering(image, image_copy)) {
		cout << "Congratulations!\nLighting successfully completed!" << endl;
		writePPM("afterCPU.ppm", image);
		writePPM("afterGPU.ppm", image_copy);
	}

	free(image), free(image_copy);
	cudaFree(cuImage);

	return 0;
}
