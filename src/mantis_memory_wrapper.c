#include <stddef.h>
#include <stdint.h>
#include "darknet.h"
#include "image.h"


#ifdef GPU
void cudaWrapper_freeHost(void* ptr){
	cudaFreeHost(ptr);
}

void cudaWrapper_free(void* ptr){
	cudaFree(ptr);
}

void cudaWrapper_malloc(void** ptr, size_t sz){
	cudaMalloc(ptr, sz);
}

image cudaWrapper_make_image_host(int w, int h, int c){
	return make_image_host(w, h, c);
}

void cudaWrapper_cp_host_to_device(void* dst_ptr, void* src_ptr, size_t size){
	cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyHostToDevice);
}

void cudaWrapper_cp_device_to_host(void* dst_ptr, void* src_ptr, size_t size){
	cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToHost);
}
#endif
