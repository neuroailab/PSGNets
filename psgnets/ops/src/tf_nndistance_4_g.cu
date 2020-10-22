#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

__global__ void NmDistance4Kernel(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i){
        const int batch=512;
        __shared__ float buf[batch*12];
        for (int i=blockIdx.x;i<b;i+=gridDim.x){
                for (int k2=0;k2<m;k2+=batch){
                        int end_k=min(m,k2+batch)-k2;
                        for (int j=threadIdx.x;j<end_k*12;j+=blockDim.x){
                                buf[j]=xyz2[(i*m+k2)*12+j];
                        }
                        __syncthreads();
                        for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
                                float x1=xyz[(i*n+j)*12+0];
                                float y1=xyz[(i*n+j)*12+1];
                                float z1=xyz[(i*n+j)*12+2];
                                float x1_1=xyz[(i*n+j)*12+3];
                                float y1_1=xyz[(i*n+j)*12+4];
                                float z1_1=xyz[(i*n+j)*12+5];
                                float x1_2=xyz[(i*n+j)*12+6];
                                float y1_2=xyz[(i*n+j)*12+7];
                                float z1_2=xyz[(i*n+j)*12+8];
                                float x1_3=xyz[(i*n+j)*12+9];
                                float y1_3=xyz[(i*n+j)*12+10];
                                float z1_3=xyz[(i*n+j)*12+11];
                                int best_i=0;
                                float best=0;
                                int end_ka=end_k-(end_k&3);
                                if (end_ka==batch){
                                        for (int k=0;k<batch;k+=4){
                                                {
                                                        float x2=buf[k*12+0]-x1;
                                                        float y2=buf[k*12+1]-y1;
                                                        float z2=buf[k*12+2]-z1;
                                                        float x2_1=buf[k*12+3]-x1_1;
                                                        float y2_1=buf[k*12+4]-y1_1;
                                                        float z2_1=buf[k*12+5]-z1_1;
                                                        float x2_2=buf[k*12+6]-x1_2;
                                                        float y2_2=buf[k*12+7]-y1_2;
                                                        float z2_2=buf[k*12+8]-z1_2;
                                                        float x2_3=buf[k*12+9]-x1_3;
                                                        float y2_3=buf[k*12+10]-y1_3;
                                                        float z2_3=buf[k*12+11]-z1_3;
                                                        float d=x2*x2+y2*y2+z2*z2+x2_1*x2_1+y2_1*y2_1+z2_1*z2_1+x2_2*x2_2+y2_2*y2_2+z2_2*z2_2+x2_3*x2_3+y2_3*y2_3+z2_3*z2_3;
                                                        if (k==0 || d<best){
                                                                best=d;
                                                                best_i=k+k2;
                                                        }
                                                }
                                                {
                                                        float x2=buf[k*12+12]-x1;
                                                        float y2=buf[k*12+13]-y1;
                                                        float z2=buf[k*12+14]-z1;
                                                        float x2_1=buf[k*12+15]-x1_1;
                                                        float y2_1=buf[k*12+16]-y1_1;
                                                        float z2_1=buf[k*12+17]-z1_1;
                                                        float x2_2=buf[k*12+18]-x1_2;
                                                        float y2_2=buf[k*12+19]-y1_2;
                                                        float z2_2=buf[k*12+20]-z1_2;
                                                        float x2_3=buf[k*12+21]-x1_3;
                                                        float y2_3=buf[k*12+22]-y1_3;
                                                        float z2_3=buf[k*12+23]-z1_3;
                                                        float d=x2*x2+y2*y2+z2*z2+x2_1*x2_1+y2_1*y2_1+z2_1*z2_1+x2_2*x2_2+y2_2*y2_2+z2_2*z2_2+x2_3*x2_3+y2_3*y2_3+z2_3*z2_3;
                                                        if (d<best){
                                                                best=d;
                                                                best_i=k+k2+1;
                                                        }
                                                }
                                                {
                                                        float x2=buf[k*12+24]-x1;
                                                        float y2=buf[k*12+25]-y1;
                                                        float z2=buf[k*12+26]-z1;
                                                        float x2_1=buf[k*12+27]-x1_1;
                                                        float y2_1=buf[k*12+28]-y1_1;
                                                        float z2_1=buf[k*12+29]-z1_1;
                                                        float x2_2=buf[k*12+30]-x1_2;
                                                        float y2_2=buf[k*12+31]-y1_2;
                                                        float z2_2=buf[k*12+32]-z1_2;
                                                        float x2_3=buf[k*12+33]-x1_3;
                                                        float y2_3=buf[k*12+34]-y1_3;
                                                        float z2_3=buf[k*12+35]-z1_3;
                                                        float d=x2*x2+y2*y2+z2*z2+x2_1*x2_1+y2_1*y2_1+z2_1*z2_1+x2_2*x2_2+y2_2*y2_2+z2_2*z2_2+x2_3*x2_3+y2_3*y2_3+z2_3*z2_3;
                                                        if (d<best){
                                                                best=d;
                                                                best_i=k+k2+2;
                                                        }
                                                }
                                                {
                                                        float x2=buf[k*12+36]-x1;
                                                        float y2=buf[k*12+37]-y1;
                                                        float z2=buf[k*12+38]-z1;
                                                        float x2_1=buf[k*12+39]-x1_1;
                                                        float y2_1=buf[k*12+40]-y1_1;
                                                        float z2_1=buf[k*12+41]-z1_1;
                                                        float x2_2=buf[k*12+42]-x1_2;
                                                        float y2_2=buf[k*12+43]-y1_2;
                                                        float z2_2=buf[k*12+44]-z1_2;
                                                        float x2_3=buf[k*12+45]-x1_3;
                                                        float y2_3=buf[k*12+46]-y1_3;
                                                        float z2_3=buf[k*12+47]-z1_3;
                                                        float d=x2*x2+y2*y2+z2*z2+x2_1*x2_1+y2_1*y2_1+z2_1*z2_1+x2_2*x2_2+y2_2*y2_2+z2_2*z2_2+x2_3*x2_3+y2_3*y2_3+z2_3*z2_3;
                                                        if (d<best){
                                                                best=d;
                                                                best_i=k+k2+3;
                                                        }
                                                }
                                        }
                                }else{
                                        for (int k=0;k<end_ka;k+=4){
                                                {
                                                        float x2=buf[k*12+0]-x1;
                                                        float y2=buf[k*12+1]-y1;
                                                        float z2=buf[k*12+2]-z1;
                                                        float x2_1=buf[k*12+3]-x1_1;
                                                        float y2_1=buf[k*12+4]-y1_1;
                                                        float z2_1=buf[k*12+5]-z1_1;
                                                        float x2_2=buf[k*12+6]-x1_2;
                                                        float y2_2=buf[k*12+7]-y1_2;
                                                        float z2_2=buf[k*12+8]-z1_2;
                                                        float x2_3=buf[k*12+9]-x1_3;
                                                        float y2_3=buf[k*12+10]-y1_3;
                                                        float z2_3=buf[k*12+11]-z1_3;
                                                        float d=x2*x2+y2*y2+z2*z2+x2_1*x2_1+y2_1*y2_1+z2_1*z2_1+x2_2*x2_2+y2_2*y2_2+z2_2*z2_2+x2_3*x2_3+y2_3*y2_3+z2_3*z2_3;
                                                        if (k==0 || d<best){
                                                                best=d;
                                                                best_i=k+k2;
                                                        }
                                                }
                                                {
                                                        float x2=buf[k*12+12]-x1;
                                                        float y2=buf[k*12+13]-y1;
                                                        float z2=buf[k*12+14]-z1;
                                                        float x2_1=buf[k*12+15]-x1_1;
                                                        float y2_1=buf[k*12+16]-y1_1;
                                                        float z2_1=buf[k*12+17]-z1_1;
                                                        float x2_2=buf[k*12+18]-x1_2;
                                                        float y2_2=buf[k*12+19]-y1_2;
                                                        float z2_2=buf[k*12+20]-z1_2;
                                                        float x2_3=buf[k*12+21]-x1_3;
                                                        float y2_3=buf[k*12+22]-y1_3;
                                                        float z2_3=buf[k*12+23]-z1_3;
                                                        float d=x2*x2+y2*y2+z2*z2+x2_1*x2_1+y2_1*y2_1+z2_1*z2_1+x2_2*x2_2+y2_2*y2_2+z2_2*z2_2+x2_3*x2_3+y2_3*y2_3+z2_3*z2_3;
                                                        if (d<best){
                                                                best=d;
                                                                best_i=k+k2+1;
                                                        }
                                                }
                                                {
                                                        float x2=buf[k*12+24]-x1;
                                                        float y2=buf[k*12+25]-y1;
                                                        float z2=buf[k*12+26]-z1;
                                                        float x2_1=buf[k*12+27]-x1_1;
                                                        float y2_1=buf[k*12+28]-y1_1;
                                                        float z2_1=buf[k*12+29]-z1_1;
                                                        float x2_2=buf[k*12+30]-x1_2;
                                                        float y2_2=buf[k*12+31]-y1_2;
                                                        float z2_2=buf[k*12+32]-z1_2;
                                                        float x2_3=buf[k*12+33]-x1_3;
                                                        float y2_3=buf[k*12+34]-y1_3;
                                                        float z2_3=buf[k*12+35]-z1_3;
                                                        float d=x2*x2+y2*y2+z2*z2+x2_1*x2_1+y2_1*y2_1+z2_1*z2_1+x2_2*x2_2+y2_2*y2_2+z2_2*z2_2+x2_3*x2_3+y2_3*y2_3+z2_3*z2_3;
                                                        if (d<best){
                                                                best=d;
                                                                best_i=k+k2+2;
                                                        }
                                                }
                                                {
                                                        float x2=buf[k*12+36]-x1;
                                                        float y2=buf[k*12+37]-y1;
                                                        float z2=buf[k*12+38]-z1;
                                                        float x2_1=buf[k*12+39]-x1_1;
                                                        float y2_1=buf[k*12+40]-y1_1;
                                                        float z2_1=buf[k*12+41]-z1_1;
                                                        float x2_2=buf[k*12+42]-x1_2;
                                                        float y2_2=buf[k*12+43]-y1_2;
                                                        float z2_2=buf[k*12+44]-z1_2;
                                                        float x2_3=buf[k*12+45]-x1_3;
                                                        float y2_3=buf[k*12+46]-y1_3;
                                                        float z2_3=buf[k*12+47]-z1_3;
                                                        float d=x2*x2+y2*y2+z2*z2+x2_1*x2_1+y2_1*y2_1+z2_1*z2_1+x2_2*x2_2+y2_2*y2_2+z2_2*z2_2+x2_3*x2_3+y2_3*y2_3+z2_3*z2_3;
                                                        if (d<best){
                                                                best=d;
                                                                best_i=k+k2+3;
                                                        }
                                                }
                                        }
                                }
                                for (int k=end_ka;k<end_k;k++){
                                        float x2=buf[k*12+0]-x1;
                                        float y2=buf[k*12+1]-y1;
                                        float z2=buf[k*12+2]-z1;
                                        float x2_1=buf[k*12+3]-x1_1;
                                        float y2_1=buf[k*12+4]-y1_1;
                                        float z2_1=buf[k*12+5]-z1_1;
                                        float x2_2=buf[k*12+6]-x1_2;
                                        float y2_2=buf[k*12+7]-y1_2;
                                        float z2_2=buf[k*12+8]-z1_2;
                                        float x2_3=buf[k*12+9]-x1_3;
                                        float y2_3=buf[k*12+10]-y1_3;
                                        float z2_3=buf[k*12+11]-z1_3;
                                        float d=x2*x2+y2*y2+z2*z2+x2_1*x2_1+y2_1*y2_1+z2_1*z2_1+x2_2*x2_2+y2_2*y2_2+z2_2*z2_2+x2_3*x2_3+y2_3*y2_3+z2_3*z2_3;
                                        if (k==0 || d<best){
                                                best=d;
                                                best_i=k+k2;
                                        }
                                }
                                if (k2==0 || result[(i*n+j)]>best){
                                        result[(i*n+j)]=best;
                                        result_i[(i*n+j)]=best_i;
                                }
                        }
                        __syncthreads();
                }
        }
}
void NmDistance4KernelLauncher(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,float * result2,int * result2_i){
        NmDistance4Kernel<<<dim3(32,16,1),512>>>(b,n,xyz,m,xyz2,result,result_i);
        NmDistance4Kernel<<<dim3(32,16,1),512>>>(b,m,xyz2,n,xyz,result2,result2_i);
}
__global__ void NmDistance4GradKernel(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,float * grad_xyz1,float * grad_xyz2){
        for (int i=blockIdx.x;i<b;i+=gridDim.x){
                for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
                        float x1=xyz1[(i*n+j)*12+0];
                        float y1=xyz1[(i*n+j)*12+1];
                        float z1=xyz1[(i*n+j)*12+2];
                        float x1_1=xyz1[(i*n+j)*12+3];
                        float y1_1=xyz1[(i*n+j)*12+4];
                        float z1_1=xyz1[(i*n+j)*12+5];
                        float x1_2=xyz1[(i*n+j)*12+6];
                        float y1_2=xyz1[(i*n+j)*12+7];
                        float z1_2=xyz1[(i*n+j)*12+8];
                        float x1_3=xyz1[(i*n+j)*12+9];
                        float y1_3=xyz1[(i*n+j)*12+10];
                        float z1_3=xyz1[(i*n+j)*12+11];
                        int j2=idx1[i*n+j];
                        float x2=xyz2[(i*m+j2)*12+0];
                        float y2=xyz2[(i*m+j2)*12+1];
                        float z2=xyz2[(i*m+j2)*12+2];
                        float x2_1=xyz2[(i*m+j2)*12+3];
                        float y2_1=xyz2[(i*m+j2)*12+4];
                        float z2_1=xyz2[(i*m+j2)*12+5];
                        float x2_2=xyz2[(i*m+j2)*12+6];
                        float y2_2=xyz2[(i*m+j2)*12+7];
                        float z2_2=xyz2[(i*m+j2)*12+8];
                        float x2_3=xyz2[(i*m+j2)*12+9];
                        float y2_3=xyz2[(i*m+j2)*12+10];
                        float z2_3=xyz2[(i*m+j2)*12+11];
                        float g=grad_dist1[i*n+j]*2;
                        atomicAdd(&(grad_xyz1[(i*n+j)*12+0]),g*(x1-x2));
                        atomicAdd(&(grad_xyz1[(i*n+j)*12+1]),g*(y1-y2));
                        atomicAdd(&(grad_xyz1[(i*n+j)*12+2]),g*(z1-z2));
                        atomicAdd(&(grad_xyz1[(i*n+j)*12+3]),g*(x1_1-x2_1));
                        atomicAdd(&(grad_xyz1[(i*n+j)*12+4]),g*(y1_1-y2_1));
                        atomicAdd(&(grad_xyz1[(i*n+j)*12+5]),g*(z1_1-z2_1));
                        atomicAdd(&(grad_xyz1[(i*n+j)*12+6]),g*(x1_2-x2_2));
                        atomicAdd(&(grad_xyz1[(i*n+j)*12+7]),g*(y1_2-y2_2));
                        atomicAdd(&(grad_xyz1[(i*n+j)*12+8]),g*(z1_2-z2_2));
                        atomicAdd(&(grad_xyz1[(i*n+j)*12+9]),g*(x1_3-x2_3));
                        atomicAdd(&(grad_xyz1[(i*n+j)*12+10]),g*(y1_3-y2_3));
                        atomicAdd(&(grad_xyz1[(i*n+j)*12+11]),g*(z1_3-z2_3));
                        atomicAdd(&(grad_xyz2[(i*m+j2)*12+0]),-(g*(x1-x2)));
                        atomicAdd(&(grad_xyz2[(i*m+j2)*12+1]),-(g*(y1-y2)));
                        atomicAdd(&(grad_xyz2[(i*m+j2)*12+2]),-(g*(z1-z2)));
                        atomicAdd(&(grad_xyz2[(i*m+j2)*12+3]),-(g*(x1_1-x2_1)));
                        atomicAdd(&(grad_xyz2[(i*m+j2)*12+4]),-(g*(y1_1-y2_1)));
                        atomicAdd(&(grad_xyz2[(i*m+j2)*12+5]),-(g*(z1_1-z2_1)));
                        atomicAdd(&(grad_xyz2[(i*m+j2)*12+6]),-(g*(x1_2-x2_2)));
                        atomicAdd(&(grad_xyz2[(i*m+j2)*12+7]),-(g*(y1_2-y2_2)));
                        atomicAdd(&(grad_xyz2[(i*m+j2)*12+8]),-(g*(z1_2-z2_2)));
                        atomicAdd(&(grad_xyz2[(i*m+j2)*12+9]),-(g*(x1_3-x2_3)));
                        atomicAdd(&(grad_xyz2[(i*m+j2)*12+10]),-(g*(y1_3-y2_3)));
                        atomicAdd(&(grad_xyz2[(i*m+j2)*12+11]),-(g*(z1_3-z2_3)));
                }
        }
}
void NmDistance4GradKernelLauncher(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,const float * grad_dist2,const int * idx2,float * grad_xyz1,float * grad_xyz2){
        cudaMemset(grad_xyz1,0,b*n*12*4);
        cudaMemset(grad_xyz2,0,b*m*12*4);
        NmDistance4GradKernel<<<dim3(1,16,1),256>>>(b,n,xyz1,m,xyz2,grad_dist1,idx1,grad_xyz1,grad_xyz2);
        NmDistance4GradKernel<<<dim3(1,16,1),256>>>(b,m,xyz2,n,xyz1,grad_dist2,idx2,grad_xyz2,grad_xyz1);
}

#endif
