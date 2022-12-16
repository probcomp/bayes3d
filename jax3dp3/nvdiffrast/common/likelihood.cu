inline __device__ int index(const int i, const int j, const int k, int width, int height, int depth) {
    return i * (height*width*4) + j * (width*4) + k*4;
}

__global__ void threedp3_likelihood(float *pos, float time, int width, int height, int depth)
{   
    int i = threadIdx.x;
    int j = blockIdx.x;
    int k = blockIdx.y;

    int filter_size = 5;

    int counter = 0;
    for(int new_j=j-filter_size; new_j <= j+filter_size; new_j++){
        for(int new_k=k-filter_size; new_k <= k+filter_size; new_k++){
            if( new_j>=0 && new_j < height && new_k >= 0 && new_k < width){
                float z = pos[index(i,new_j,new_k, width, height, depth) + 2];
                if(z > 0){
                    counter += 1;
                }
            }
        }
    }
    int idx = index(i,j,k, width, height, depth);
    pos[idx] = counter;
}
