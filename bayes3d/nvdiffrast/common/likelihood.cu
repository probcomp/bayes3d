inline __device__ int index(const int i, const int j, const int k, int width, int height, int depth) {
    return i * (height*width*4) + j * (width*4) + k*4;
}

__global__ void threedp3_likelihood(float *pos, float *latent_points, float *likelihood, float *obs_image, float r, int width, int height, int depth)
{   
    int i_0 = threadIdx.x;
    int j = blockIdx.x;
    int k = blockIdx.y;
    int block = blockIdx.z;

    int i = i_0 + 1024*block;
    if (i >= depth){
        return;
    }

    int filter_size = 3;
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
    float log_score =  log(outlier_prob + (1 - outlier_prob) / latent_points[i] * counter / (4.0/3.0 * 3.1415 * powf(r, 3)));
    if(latent_points[i] > 0 & z_o > 0)
    {
        pos[idx+3] = log_score;
    }else{
        pos[idx+3] = 0.0;
    }
    return;
}
