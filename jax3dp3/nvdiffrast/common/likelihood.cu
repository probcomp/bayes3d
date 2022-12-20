inline __device__ int index(const int i, const int j, const int k, int width, int height, int depth) {
    return i * (height*width*4) + j * (width*4) + k*4;
}

__global__ void threedp3_likelihood(float *pos, float *obs_image, float r, int width, int height, int depth)
{   
    int i_0 = threadIdx.x;
    int i_multiplier = threadIdx.y;
    int i = i_0 * i_multiplier;
    int j = blockIdx.x;
    int k = blockIdx.y;

    int filter_size = 5;
    float outlier_prob = 0.01;

    float x_o = obs_image[index(0,j,k, width, height, depth) + 0];
    float y_o = obs_image[index(0,j,k, width, height, depth) + 1];
    float z_o = obs_image[index(0,j,k, width, height, depth) + 2];

    int counter = 0;
    for(int new_j=j-filter_size; new_j <= j+filter_size; new_j++){
        for(int new_k=k-filter_size; new_k <= k+filter_size; new_k++){
            if( new_j>=0 && new_j < height && new_k >= 0 && new_k < width){
                float x = pos[index(i,new_j,new_k, width, height, depth) + 0];
                float y = pos[index(i,new_j,new_k, width, height, depth) + 1];
                float z = pos[index(i,new_j,new_k, width, height, depth) + 2];

                if(z_o > 0 && z > 0){
                    float distance = sqrt(
                        powf(x-x_o, 2) +
                        powf(y-y_o, 2) +
                        powf(z-z_o, 2) 
                    );
                    if (distance < r){
                        counter += 1;
                    }
                }
            }
        }
    }
    
    int idx = index(i,j,k, width, height, depth);
    pos[idx+3] = counter;
}
