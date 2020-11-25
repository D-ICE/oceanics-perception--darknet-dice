#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <stdint.h>

#include "im2col.h"
#include "dark_cuda.h"

#include <stdio.h>
#include <assert.h>

__global__ void mat_to_image_gpu_kernel(const int n, const float coef, const int w, const int h, const int c, const int step, unsigned char *mat_data_gpu, float* image_data_gpu){

    int index = blockIdx.x*blockDim.x + threadIdx.x;

    int k,y,x;

    for (; index < n; index += blockDim.x*gridDim.x){

        k = (index / (w*h));
        y = (index % (w*h)) / w;
        x = (index % (w*h)) % w;

        image_data_gpu[index] = ((float)mat_data_gpu[y*step + x*c + k])*coef;
      }
}

void mat_to_image_gpu(const int w, const int h, const int c, const int step, unsigned char *mat_data_gpu, float* image_data_gpu){

  int num_kernels = w*h*c;
  mat_to_image_gpu_kernel <<<(num_kernels + BLOCK - 1) / BLOCK,
      BLOCK, 0, get_cuda_stream() >>> (num_kernels, 1.0f/255.0f, w, h, c, step, mat_data_gpu, image_data_gpu);
  CHECK_CUDA(cudaPeekAtLastError());
}






__global__ void letterbox_image_gpu_kernel(const int n, const int src_w, const int src_h, const int dst_w, const int dst_h, const int channels, const float w_scale, const float h_scale, const int new_w, const int new_h, float* src_img, float* dst_img){

	int index = blockIdx.x*blockDim.x + threadIdx.x;

	int k,y,x;
	float val, sx, dx, sy, dy;
	int ix, iy;

	int x_inter, y_inter; 


	for (; index < n; index += blockDim.x*gridDim.x){
	
		k = (index / (dst_h*dst_w));
		y = (index % (dst_h*dst_w)) / dst_w;
        x = (index % (dst_h*dst_w)) % dst_w;

		
		x_inter = x - (dst_w - new_w) / 2;
		y_inter = y - (dst_h - new_h) / 2;
		
		dst_img[index] = 0.5f;

		if(x_inter>=0 && y_inter>=0 && x_inter < new_w && y_inter < new_h){
		
			sy = y_inter*h_scale;
			iy = (int) sy;
			dy = sy - iy;
			
			if(x_inter == new_w-1 || src_w == 1){
				val = src_img[(src_w-1) + iy*src_w + k*src_w*src_h];

				//printf("[GPU] DST x,y,k=(%d %d %d) <- (1-dy=%f) * SRC (src_w-1),iy,k=(%d %d %d) v = %f\n", x, y, k, dy, (src_w-1), iy, k, val);
			}
			else{
				sx = x_inter*w_scale;
		        ix = (int) sx;
		        dx = sx - ix;
				val = (1 - dx) * src_img[ix + iy*src_w + k*src_w*src_h]
					+ dx * src_img[ix+1 + iy*src_w + k*src_w*src_h];

				//printf("[GPU] DST x,y,k=(%d %d %d) <- (1-dy=%f) * ( (1 - dx=%f) SRC ix,iy,k=(%d %d %d) + dx=%f * SRC ix+1,iy,k=(%d %d %d) v = %f\n", x, y, k, dy, dx, ix, iy, k, dx, ix+1, iy, k, val);
			}
			
			dst_img[index] = (1-dy) * val;

			if(y_inter != new_h-1 && src_h != 1){
				if(x_inter == new_w-1 || src_w == 1){
					val = src_img[(src_w-1) + (iy+1)*src_w + k*src_w*src_h];
					//printf("[GPU] PLUS <- dy=%f * SRC (src_w-1),iy+1,k=(%d %d %d) v = %f\n", dy, (src_w-1), iy+1, k, val);
				}
				else{
					sx = x_inter*w_scale;
					ix = (int) sx;
					dx = sx - ix;
					val = (1 - dx) * src_img[ix + (iy+1)*src_w + k*src_w*src_h] + dx * src_img[ix+1 + (iy+1)*src_w + k*src_w*src_h];
					
					//printf("[GPU] PLUS <- (1-dx)=%f * ( SRC ix,iy+1,k=(%d %d %d)  +  dx=%f * SRC ix+1,iy+1,k=(%d %d %d))  v = %f\n", (1-dx), ix, iy+1, k, dx, ix+1 ,iy+1, k, val);
					
				}
				
				dst_img[index] += dy * val;
			}
		}
	}
}


void letterbox_image_gpu(const int src_w, const int src_h, const int dst_w, const int dst_h, const int c, float* src_img, float* dst_img){

	int new_w = src_w;
    int new_h = src_h;
    if (((float)dst_w / src_w) < ((float)dst_h / src_h)) {
        new_w = dst_w;
        new_h = (src_h * dst_w) / src_w;
    }
    else {
        new_h = dst_h;
        new_w = (src_w * dst_h) / src_h;
    }

	float w_scale = (float)(src_w - 1) / (new_w - 1);
    float h_scale = (float)(src_h - 1) / (new_h - 1);


	int num_kernels = dst_w*dst_h*c;
	letterbox_image_gpu_kernel <<<(num_kernels + BLOCK - 1) / BLOCK,
	  BLOCK, 0, get_cuda_stream() >>> 
		(num_kernels, src_w, src_h, dst_w, dst_h, c, w_scale, h_scale, new_w, new_h, src_img, dst_img);
	CHECK_CUDA(cudaPeekAtLastError());

}











