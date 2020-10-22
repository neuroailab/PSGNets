#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("NnDistance4")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Output("dist1: float32")
    .Output("idx1: int32")
    .Output("dist2: float32")
    .Output("idx2: int32");
REGISTER_OP("NnDistance4Grad")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Input("grad_dist1: float32")
    .Input("idx1: int32")
    .Input("grad_dist2: float32")
    .Input("idx2: int32")
    .Output("grad_xyz1: float32")
    .Output("grad_xyz2: float32");
using namespace tensorflow;

static void nnsearch(int b,int n,int m,const float * xyz1,const float * xyz2,float * dist,int * idx){
        for (int i=0;i<b;i++){
                for (int j=0;j<n;j++){
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
                        double best=0;
                        int besti=0;
                        for (int k=0;k<m;k++){
                                float x2=xyz2[(i*m+k)*12+0]-x1;
                                float y2=xyz2[(i*m+k)*12+1]-y1;
                                float z2=xyz2[(i*m+k)*12+2]-z1;
                                float x2_1=xyz2[(i*m+k)*12+3]-x1_1;
                                float y2_1=xyz2[(i*m+k)*12+4]-y1_1;
                                float z2_1=xyz2[(i*m+k)*12+5]-z1_1;
                                float x2_2=xyz2[(i*m+k)*12+6]-x1_2;
                                float y2_2=xyz2[(i*m+k)*12+7]-y1_2;
                                float z2_2=xyz2[(i*m+k)*12+8]-z1_2;
                                float x2_3=xyz2[(i*m+k)*12+9]-x1_3;
                                float y2_3=xyz2[(i*m+k)*12+10]-y1_3;
                                float z2_3=xyz2[(i*m+k)*12+11]-z1_3;
                                double d=x2*x2+y2*y2+z2*z2+x2_1*x2_1+y2_1*y2_1+z2_1*z2_1+x2_2*x2_2+y2_2*y2_2+z2_2*z2_2+x2_3*x2_3+y2_3*y2_3+z2_3*z2_3;
                                if (k==0 || d<best){
                                        best=d;
                                        besti=k;
                                }
                        }
                        dist[i*n+j]=best;
                        idx[i*n+j]=besti;
                }
        }
}

class NnDistance4Op : public OpKernel{
        public:
                explicit NnDistance4Op(OpKernelConstruction* context):OpKernel(context){}
                void Compute(OpKernelContext * context)override{
                        const Tensor& xyz1_tensor=context->input(0);
                        const Tensor& xyz2_tensor=context->input(1);
                        OP_REQUIRES(context,xyz1_tensor.dims()==3,errors::InvalidArgument("NnDistance requires xyz1 be of shape (batch,#points,12)"));
                        OP_REQUIRES(context,xyz1_tensor.shape().dim_size(2)==12,errors::InvalidArgument("NnDistance only accepts 12d point set xyz1"));
                        int b=xyz1_tensor.shape().dim_size(0);
                        int n=xyz1_tensor.shape().dim_size(1);
                        OP_REQUIRES(context,xyz2_tensor.dims()==3,errors::InvalidArgument("NnDistance requires xyz2 be of shape (batch,#points,12)"));
                        OP_REQUIRES(context,xyz2_tensor.shape().dim_size(2)==12,errors::InvalidArgument("NnDistance only accepts 12d point set xyz2"));
                        int m=xyz2_tensor.shape().dim_size(1);
                        OP_REQUIRES(context,xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("NnDistance expects xyz1 and xyz2 have same batch size"));
                        auto xyz1_flat=xyz1_tensor.flat<float>();
                        const float * xyz1=&xyz1_flat(0);
                        auto xyz2_flat=xyz2_tensor.flat<float>();
                        const float * xyz2=&xyz2_flat(0);
                        Tensor * dist1_tensor=NULL;
                        Tensor * idx1_tensor=NULL;
                        Tensor * dist2_tensor=NULL;
                        Tensor * idx2_tensor=NULL;
                        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n},&dist1_tensor));
                        OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,n},&idx1_tensor));
                        auto dist1_flat=dist1_tensor->flat<float>();
                        auto idx1_flat=idx1_tensor->flat<int>();
                        OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape{b,m},&dist2_tensor));
                        OP_REQUIRES_OK(context,context->allocate_output(3,TensorShape{b,m},&idx2_tensor));
                        auto dist2_flat=dist2_tensor->flat<float>();
                        auto idx2_flat=idx2_tensor->flat<int>();
                        float * dist1=&(dist1_flat(0));
                        int * idx1=&(idx1_flat(0));
                        float * dist2=&(dist2_flat(0));
                        int * idx2=&(idx2_flat(0));
                        nnsearch(b,n,m,xyz1,xyz2,dist1,idx1);
                        nnsearch(b,m,n,xyz2,xyz1,dist2,idx2);
                }
};
REGISTER_KERNEL_BUILDER(Name("NnDistance4").Device(DEVICE_CPU), NnDistance4Op);
class NnDistance4GradOp : public OpKernel{
        public:
                explicit NnDistance4GradOp(OpKernelConstruction* context):OpKernel(context){}
                void Compute(OpKernelContext * context)override{
                        const Tensor& xyz1_tensor=context->input(0);
                        const Tensor& xyz2_tensor=context->input(1);
                        const Tensor& grad_dist1_tensor=context->input(2);
                        const Tensor& idx1_tensor=context->input(3);
                        const Tensor& grad_dist2_tensor=context->input(4);
                        const Tensor& idx2_tensor=context->input(5);
                        OP_REQUIRES(context,xyz1_tensor.dims()==3,errors::InvalidArgument("NnDistanceGrad requires xyz1 be of shape (batch,#points,12)"));
                        OP_REQUIRES(context,xyz1_tensor.shape().dim_size(2)==12,errors::InvalidArgument("NnDistanceGrad only accepts 12d point set xyz1"));
                        int b=xyz1_tensor.shape().dim_size(0);
                        int n=xyz1_tensor.shape().dim_size(1);
                        OP_REQUIRES(context,xyz2_tensor.dims()==3,errors::InvalidArgument("NnDistanceGrad requires xyz2 be of shape (batch,#points,12)"));
                        OP_REQUIRES(context,xyz2_tensor.shape().dim_size(2)==12,errors::InvalidArgument("NnDistanceGrad only accepts 12d point set xyz2"));
                        int m=xyz2_tensor.shape().dim_size(1);
                        OP_REQUIRES(context,xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("NnDistanceGrad expects xyz1 and xyz2 have same batch size"));
                        OP_REQUIRES(context,grad_dist1_tensor.shape()==(TensorShape{b,n}),errors::InvalidArgument("NnDistanceGrad requires grad_dist1 be of shape(batch,#points)"));
                        OP_REQUIRES(context,idx1_tensor.shape()==(TensorShape{b,n}),errors::InvalidArgument("NnDistanceGrad requires idx1 be of shape(batch,#points)"));
                        OP_REQUIRES(context,grad_dist2_tensor.shape()==(TensorShape{b,m}),errors::InvalidArgument("NnDistanceGrad requires grad_dist2 be of shape(batch,#points)"));
                        OP_REQUIRES(context,idx2_tensor.shape()==(TensorShape{b,m}),errors::InvalidArgument("NnDistanceGrad requires idx2 be of shape(batch,#points)"));
                        auto xyz1_flat=xyz1_tensor.flat<float>();
                        const float * xyz1=&xyz1_flat(0);
                        auto xyz2_flat=xyz2_tensor.flat<float>();
                        const float * xyz2=&xyz2_flat(0);
                        auto idx1_flat=idx1_tensor.flat<int>();
                        const int * idx1=&idx1_flat(0);
                        auto idx2_flat=idx2_tensor.flat<int>();
                        const int * idx2=&idx2_flat(0);
                        auto grad_dist1_flat=grad_dist1_tensor.flat<float>();
                        const float * grad_dist1=&grad_dist1_flat(0);
                        auto grad_dist2_flat=grad_dist2_tensor.flat<float>();
                        const float * grad_dist2=&grad_dist2_flat(0);
                        Tensor * grad_xyz1_tensor=NULL;
                        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,12},&grad_xyz1_tensor));
                        Tensor * grad_xyz2_tensor=NULL;
                        OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,m,12},&grad_xyz2_tensor));
                        auto grad_xyz1_flat=grad_xyz1_tensor->flat<float>();
                        float * grad_xyz1=&grad_xyz1_flat(0);
                        auto grad_xyz2_flat=grad_xyz2_tensor->flat<float>();
                        float * grad_xyz2=&grad_xyz2_flat(0);
                        for (int i=0;i<b*n*12;i++)
                                grad_xyz1[i]=0;
                        for (int i=0;i<b*m*12;i++)
                                grad_xyz2[i]=0;
                        for (int i=0;i<b;i++){
                                for (int j=0;j<n;j++){
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
                                        grad_xyz1[(i*n+j)*12+0]+=g*(x1-x2);
                                        grad_xyz1[(i*n+j)*12+1]+=g*(y1-y2);
                                        grad_xyz1[(i*n+j)*12+2]+=g*(z1-z2);
                                        grad_xyz1[(i*n+j)*12+3]+=g*(x1_1-x2_1);
                                        grad_xyz1[(i*n+j)*12+4]+=g*(y1_1-y2_1);
                                        grad_xyz1[(i*n+j)*12+5]+=g*(z1_1-z2_1);
                                        grad_xyz1[(i*n+j)*12+6]+=g*(x1_2-x2_2);
                                        grad_xyz1[(i*n+j)*12+7]+=g*(y1_2-y2_2);
                                        grad_xyz1[(i*n+j)*12+8]+=g*(z1_2-z2_2);
                                        grad_xyz1[(i*n+j)*12+9]+=g*(x1_3-x2_3);
                                        grad_xyz1[(i*n+j)*12+10]+=g*(y1_3-y2_3);
                                        grad_xyz1[(i*n+j)*12+11]+=g*(z1_3-z2_3);
                                        grad_xyz2[(i*m+j2)*12+0]-=(g*(x1-x2));
                                        grad_xyz2[(i*m+j2)*12+1]-=(g*(y1-y2));
                                        grad_xyz2[(i*m+j2)*12+2]-=(g*(z1-z2));
                                        grad_xyz2[(i*m+j2)*12+3]-=(g*(x1_1-x2_1));
                                        grad_xyz2[(i*m+j2)*12+4]-=(g*(y1_1-y2_1));
                                        grad_xyz2[(i*m+j2)*12+5]-=(g*(z1_1-z2_1));
                                        grad_xyz2[(i*m+j2)*12+6]-=(g*(x1_2-x2_2));
                                        grad_xyz2[(i*m+j2)*12+7]-=(g*(y1_2-y2_2));
                                        grad_xyz2[(i*m+j2)*12+8]-=(g*(z1_2-z2_2));
                                        grad_xyz2[(i*m+j2)*12+9]-=(g*(x1_3-x2_3));
                                        grad_xyz2[(i*m+j2)*12+10]-=(g*(y1_3-y2_3));
                                        grad_xyz2[(i*m+j2)*12+11]-=(g*(z1_3-z2_3));
                                }
                                for (int j=0;j<m;j++){
                                        float x1=xyz2[(i*m+j)*12+0];
                                        float y1=xyz2[(i*m+j)*12+1];
                                        float z1=xyz2[(i*m+j)*12+2];
                                        float x1_1=xyz2[(i*m+j)*12+3];
                                        float y1_1=xyz2[(i*m+j)*12+4];
                                        float z1_1=xyz2[(i*m+j)*12+5];
                                        float x1_2=xyz2[(i*m+j)*12+6];
                                        float y1_2=xyz2[(i*m+j)*12+7];
                                        float z1_2=xyz2[(i*m+j)*12+8];
                                        float x1_3=xyz2[(i*m+j)*12+9];
                                        float y1_3=xyz2[(i*m+j)*12+10];
                                        float z1_3=xyz2[(i*m+j)*12+11];
                                        int j2=idx2[i*m+j];
                                        float x2=xyz1[(i*n+j2)*12+0];
                                        float y2=xyz1[(i*n+j2)*12+1];
                                        float z2=xyz1[(i*n+j2)*12+2];
                                        float x2_1=xyz1[(i*n+j2)*12+3];
                                        float y2_1=xyz1[(i*n+j2)*12+4];
                                        float z2_1=xyz1[(i*n+j2)*12+5];
                                        float x2_2=xyz1[(i*n+j2)*12+6];
                                        float y2_2=xyz1[(i*n+j2)*12+7];
                                        float z2_2=xyz1[(i*n+j2)*12+8];
                                        float x2_3=xyz1[(i*n+j2)*12+9];
                                        float y2_3=xyz1[(i*n+j2)*12+10];
                                        float z2_3=xyz1[(i*n+j2)*12+11];
                                        float g=grad_dist2[i*m+j]*2;
                                        grad_xyz2[(i*m+j)*12+0]+=g*(x1-x2);
                                        grad_xyz2[(i*m+j)*12+1]+=g*(y1-y2);
                                        grad_xyz2[(i*m+j)*12+2]+=g*(z1-z2);
                                        grad_xyz2[(i*m+j)*12+3]+=g*(x1_1-x2_1);
                                        grad_xyz2[(i*m+j)*12+4]+=g*(y1_1-y2_1);
                                        grad_xyz2[(i*m+j)*12+5]+=g*(z1_1-z2_1);
                                        grad_xyz2[(i*m+j)*12+6]+=g*(x1_2-x2_2);
                                        grad_xyz2[(i*m+j)*12+7]+=g*(y1_2-y2_2);
                                        grad_xyz2[(i*m+j)*12+8]+=g*(z1_2-z2_2);
                                        grad_xyz2[(i*m+j)*12+9]+=g*(x1_3-x2_3);
                                        grad_xyz2[(i*m+j)*12+10]+=g*(y1_3-y2_3);
                                        grad_xyz2[(i*m+j)*12+11]+=g*(z1_3-z2_3);
                                        grad_xyz1[(i*n+j2)*12+0]-=(g*(x1-x2));
                                        grad_xyz1[(i*n+j2)*12+1]-=(g*(y1-y2));
                                        grad_xyz1[(i*n+j2)*12+2]-=(g*(z1-z2));
                                        grad_xyz1[(i*n+j2)*12+3]-=(g*(x1_1-x2_1));
                                        grad_xyz1[(i*n+j2)*12+4]-=(g*(y1_1-y2_1));
                                        grad_xyz1[(i*n+j2)*12+5]-=(g*(z1_1-z2_1));
                                        grad_xyz1[(i*n+j2)*12+6]-=(g*(x1_2-x2_2));
                                        grad_xyz1[(i*n+j2)*12+7]-=(g*(y1_2-y2_2));
                                        grad_xyz1[(i*n+j2)*12+8]-=(g*(z1_2-z2_2));
                                        grad_xyz1[(i*n+j2)*12+9]-=(g*(x1_3-x2_3));
                                        grad_xyz1[(i*n+j2)*12+10]-=(g*(y1_3-y2_3));
                                        grad_xyz1[(i*n+j2)*12+11]-=(g*(z1_3-z2_3));
                                }
                        }
                }
};
REGISTER_KERNEL_BUILDER(Name("NnDistance4Grad").Device(DEVICE_CPU), NnDistance4GradOp);

void NmDistance4KernelLauncher(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,float * result2,int * result2_i);
class NnDistance4GpuOp : public OpKernel{
        public:
                explicit NnDistance4GpuOp(OpKernelConstruction* context):OpKernel(context){}
                void Compute(OpKernelContext * context)override{
                        const Tensor& xyz1_tensor=context->input(0);
                        const Tensor& xyz2_tensor=context->input(1);
                        OP_REQUIRES(context,xyz1_tensor.dims()==3,errors::InvalidArgument("NnDistance requires xyz1 be of shape (batch,#points,12)"));
                        OP_REQUIRES(context,xyz1_tensor.shape().dim_size(2)==12,errors::InvalidArgument("NnDistance only accepts 12d point set xyz1"));
                        int b=xyz1_tensor.shape().dim_size(0);
                        int n=xyz1_tensor.shape().dim_size(1);
                        OP_REQUIRES(context,xyz2_tensor.dims()==3,errors::InvalidArgument("NnDistance requires xyz2 be of shape (batch,#points,12)"));
                        OP_REQUIRES(context,xyz2_tensor.shape().dim_size(2)==12,errors::InvalidArgument("NnDistance only accepts 12d point set xyz2"));
                        int m=xyz2_tensor.shape().dim_size(1);
                        OP_REQUIRES(context,xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("NnDistance expects xyz1 and xyz2 have same batch size"));
                        auto xyz1_flat=xyz1_tensor.flat<float>();
                        const float * xyz1=&xyz1_flat(0);
                        auto xyz2_flat=xyz2_tensor.flat<float>();
                        const float * xyz2=&xyz2_flat(0);
                        Tensor * dist1_tensor=NULL;
                        Tensor * idx1_tensor=NULL;
                        Tensor * dist2_tensor=NULL;
                        Tensor * idx2_tensor=NULL;
                        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n},&dist1_tensor));
                        OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,n},&idx1_tensor));
                        auto dist1_flat=dist1_tensor->flat<float>();
                        auto idx1_flat=idx1_tensor->flat<int>();
                        OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape{b,m},&dist2_tensor));
                        OP_REQUIRES_OK(context,context->allocate_output(3,TensorShape{b,m},&idx2_tensor));
                        auto dist2_flat=dist2_tensor->flat<float>();
                        auto idx2_flat=idx2_tensor->flat<int>();
                        float * dist1=&(dist1_flat(0));
                        int * idx1=&(idx1_flat(0));
                        float * dist2=&(dist2_flat(0));
                        int * idx2=&(idx2_flat(0));
                        NmDistance4KernelLauncher(b,n,xyz1,m,xyz2,dist1,idx1,dist2,idx2);
                }
};
REGISTER_KERNEL_BUILDER(Name("NnDistance4").Device(DEVICE_GPU), NnDistance4GpuOp);

void NmDistance4GradKernelLauncher(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,const float * grad_dist2,const int * idx2,float * grad_xyz1,float * grad_xyz2);
class NnDistance4GradGpuOp : public OpKernel{
        public:
                explicit NnDistance4GradGpuOp(OpKernelConstruction* context):OpKernel(context){}
                void Compute(OpKernelContext * context)override{
                        const Tensor& xyz1_tensor=context->input(0);
                        const Tensor& xyz2_tensor=context->input(1);
                        const Tensor& grad_dist1_tensor=context->input(2);
                        const Tensor& idx1_tensor=context->input(3);
                        const Tensor& grad_dist2_tensor=context->input(4);
                        const Tensor& idx2_tensor=context->input(5);
                        OP_REQUIRES(context,xyz1_tensor.dims()==3,errors::InvalidArgument("NnDistanceGrad requires xyz1 be of shape (batch,#points,12)"));
                        OP_REQUIRES(context,xyz1_tensor.shape().dim_size(2)==12,errors::InvalidArgument("NnDistanceGrad only accepts 12d point set xyz1"));
                        int b=xyz1_tensor.shape().dim_size(0);
                        int n=xyz1_tensor.shape().dim_size(1);
                        OP_REQUIRES(context,xyz2_tensor.dims()==3,errors::InvalidArgument("NnDistanceGrad requires xyz2 be of shape (batch,#points,12)"));
                        OP_REQUIRES(context,xyz2_tensor.shape().dim_size(2)==12,errors::InvalidArgument("NnDistanceGrad only accepts 12d point set xyz2"));
                        int m=xyz2_tensor.shape().dim_size(1);
                        OP_REQUIRES(context,xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("NnDistanceGrad expects xyz1 and xyz2 have same batch size"));
                        OP_REQUIRES(context,grad_dist1_tensor.shape()==(TensorShape{b,n}),errors::InvalidArgument("NnDistanceGrad requires grad_dist1 be of shape(batch,#points)"));
                        OP_REQUIRES(context,idx1_tensor.shape()==(TensorShape{b,n}),errors::InvalidArgument("NnDistanceGrad requires idx1 be of shape(batch,#points)"));
                        OP_REQUIRES(context,grad_dist2_tensor.shape()==(TensorShape{b,m}),errors::InvalidArgument("NnDistanceGrad requires grad_dist2 be of shape(batch,#points)"));
                        OP_REQUIRES(context,idx2_tensor.shape()==(TensorShape{b,m}),errors::InvalidArgument("NnDistanceGrad requires idx2 be of shape(batch,#points)"));
                        auto xyz1_flat=xyz1_tensor.flat<float>();
                        const float * xyz1=&xyz1_flat(0);
                        auto xyz2_flat=xyz2_tensor.flat<float>();
                        const float * xyz2=&xyz2_flat(0);
                        auto idx1_flat=idx1_tensor.flat<int>();
                        const int * idx1=&idx1_flat(0);
                        auto idx2_flat=idx2_tensor.flat<int>();
                        const int * idx2=&idx2_flat(0);
                        auto grad_dist1_flat=grad_dist1_tensor.flat<float>();
                        const float * grad_dist1=&grad_dist1_flat(0);
                        auto grad_dist2_flat=grad_dist2_tensor.flat<float>();
                        const float * grad_dist2=&grad_dist2_flat(0);
                        Tensor * grad_xyz1_tensor=NULL;
                        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,12},&grad_xyz1_tensor));
                        Tensor * grad_xyz2_tensor=NULL;
                        OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,m,12},&grad_xyz2_tensor));
                        auto grad_xyz1_flat=grad_xyz1_tensor->flat<float>();
                        float * grad_xyz1=&grad_xyz1_flat(0);
                        auto grad_xyz2_flat=grad_xyz2_tensor->flat<float>();
                        float * grad_xyz2=&grad_xyz2_flat(0);
                        NmDistance4GradKernelLauncher(b,n,xyz1,m,xyz2,grad_dist1,idx1,grad_dist2,idx2,grad_xyz1,grad_xyz2);
                }
};
REGISTER_KERNEL_BUILDER(Name("NnDistance4Grad").Device(DEVICE_GPU), NnDistance4GradGpuOp);
