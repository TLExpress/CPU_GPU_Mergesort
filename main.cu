// System include
#include <stdlib.h> // For fflush and rand
#include <stdint.h> // For uint64_t
#include <stdio.h> // Standard IO library
#include <time.h> // For srand()

// For the CUDA runtime routines
#include <cuda_runtime.h>
#include <helper_cuda.h>

//Functions for CUDA mergesort
#include "mergesort.h"

double* genRand(uint64_t count)
{
	srand((uint64_t)time);
	double* out = new double[count] {0};

	for (uint64_t c = 0; c < count; c++)
	{
		uint64_t u = rand() * rand() * rand();
		if (u == 0.0)
		{
			c--;
			continue;
		}
		*(out+c) =  (double)((u % 3276700000) / 100000.0);
	}
	return out;
}

void sortTest(uint64_t count)
{
	double* sam = nullptr;
	sam = genRand(count);
	double* in = new double[count];
	memcpy_s(in, count, sam, count);
	fprintf(stderr, "Test count: %llu\n", count);
	fprintf(stderr, "All CPU mode\n");
	Sleep(500);
	COORPmergesort(in, count, 0.0);
	Sleep(3500);
	system("cls");
	memcpy_s(in, count, sam, count);
	fprintf(stderr, "Test count: %llu\n", count);
	fprintf(stderr, "Cooperative mode, 80%% CPU, 20%% GPU\n");
	Sleep(500);
	COORPmergesort(in, count, 0.2);
	Sleep(3500);
	system("cls");
	memcpy_s(in, count, sam, count);
	fprintf(stderr, "Test count: %llu\n", count);
	fprintf(stderr, "Cooperative mode, 60%% CPU, 40%% GPU\n");
	Sleep(500);
	COORPmergesort(in, count, 0.4);
	Sleep(3500);
	system("cls");
	memcpy_s(in, count, sam, count);
	fprintf(stderr, "Test count: %llu\n", count);
	fprintf(stderr, "Cooperative mode, 50%% CPU, 50%% GPU\n");
	Sleep(500);
	COORPmergesort(in, count, 0.5);
	Sleep(3500);
	system("cls");
	memcpy_s(in, count, sam, count);
	fprintf(stderr, "Test count: %llu\n", count);
	fprintf(stderr, "Cooperative mode, 40%% CPU, 60%% GPU\n");
	Sleep(500);
	COORPmergesort(in, count, 0.6);
	Sleep(3500);
	system("cls");
	memcpy_s(in, count, sam, count);
	fprintf(stderr, "Test count: %llu\n", count);
	fprintf(stderr, "Cooperative mode, 20%% CPU, 80%% GPU\n");
	Sleep(500);
	COORPmergesort(in, count, 0.8);
	Sleep(3500);
	system("cls");
	memcpy_s(in, count, sam, count);
	fprintf(stderr, "Test count: %llu\n", count);
	fprintf(stderr, "All GPU mode\n");
	Sleep(500);
	COORPmergesort(in, count, 1.0);
	Sleep(3500);
	system("cls");
	delete[] in;
	delete[] sam;
}
 

int main(int argc, char** argv)
{
/*
	// get the number of the inputs
	uint64_t count;
	printf("Input number:\n");
	scanf("%llu", &count);
	fflush(stdin);

	// allocate and check required memory from main memory for inputs
	size_t size = count * sizeof(double);
	double* input = (double*)malloc(size); // Alloc memory to store the unput
	if (input == NULL)
	{
		fprintf(stderr, "Failed to allocate host memory!\n");
		exit(EXIT_FAILURE);
	}
#ifdef _DEBUG
	fprintf(stderr,"\nMalloc OK!\n");
#endif

	// get input
	printf("Inputs:\n");
	for (int c0 = 0; c0 < count; c0++)
	{
		scanf("%lf", input + c0); 
		fflush(stdin);
	}

	// get sorting rate (1.0 for 100% device, 0 for 100% host)
	double rate = 0.0;
	printf("Rate:\n");
	scanf("%lf", &rate);
	fflush(stdin);

	// Do mergesort
	COORPmergesort(input, count, rate);
	
	// output the result
	printf("\nOutputs:\n");
	for (int c0 = 0; c0 < count; c0++)
		printf("%.5lf\n", *(input + c0));
	
	//release the memory of inputs
	free(input);
	return 0;

	*/
	fprintf(stderr, "CPU GPU cooperative merge sort test, 2020.\n");
	fprintf(stderr, "Made by TLExpress\n\n");
	Sleep(4000);
	system("cls");
	while (1)
	{
		sortTest(1024UL);
		sortTest(2048UL);
		sortTest(4096UL);
		sortTest(8192UL);
		sortTest(16384UL);
		sortTest(32768UL);
		sortTest(65536UL);
		sortTest(131072UL);
		//sortTest(262144UL);
		//sortTest(524288UL);
	}
}