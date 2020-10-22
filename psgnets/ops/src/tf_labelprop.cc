#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "graphs.h"

using namespace tensorflow;

// Propagates labels for each feature (h,w) in an input feature map based on
// its adjacency with other features (h',w'). Runs for num_steps then returns
// the integer labels for each feature in a [B,H,W] tensor. These will be in
// the range [0,BHW) but typically much smaller as label propagation
// tends to converge within ~5 steps to an order of magnitude fewer labels.

REGISTER_OP("LabelProp")
    .Attr("num_steps: int") // num steps to run op
    .Attr("defensive: bool = false") // whether to do defensive labelprop
    .Input("edges: bool") // connectivity matrix of shape [B,HW,(2k+1)**2]
    .Input("size: int32") // [H,W]
    .Output("labels: int32") // [B,HW] labels in increasing order
    .Output("num_segments: int32") // [B] number of segments (unique labels) per example
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	    c->set_output(0, c->Matrix(c->Dim(c->input(0),0), c->Dim(c->input(0),1)));
	    c->set_output(1, c->Vector(c->Dim(c->input(0),0)));
	    return Status::OK();
	});

static void assignLabels(int B, int H, int W, int k, int num_steps, bool defensive, const bool *edges, int *labels, int *num_segments); // declaration

class LabelPropOp : public OpKernel{
private:
    int num_steps_;
    bool defensive_;
    int H; int W;
public:
    explicit LabelPropOp(OpKernelConstruction *context):OpKernel(context){
	OP_REQUIRES_OK(context, context->GetAttr("num_steps", &num_steps_));
	OP_REQUIRES_OK(context, context->GetAttr("defensive", &defensive_));
    }
    void Compute(OpKernelContext *context) override {
	// edges input
	const Tensor &edges_tensor = context->input(0);
	OP_REQUIRES(context, edges_tensor.dims()==3, errors::InvalidArgument("LabelProp requires edges of shape (batch, H*W, (2k+1)**2"));
	int B = edges_tensor.shape().dim_size(0);
	int k = edges_tensor.shape().dim_size(2);
	k = int((std::sqrt(k)-1) / 2); // kernel half width

	auto edges_flat = edges_tensor.flat<bool>();
	const bool *edges = &edges_flat(0); // input reference

	// size input
	const Tensor &size_tensor = context->input(1);
	auto size_flat = size_tensor.flat<int>();
	const int *size = &size_flat(0);
	H = size[0]; W = size[1];
	OP_REQUIRES(context, edges_tensor.shape().dim_size(1)==(H*W), errors::InvalidArgument("LabelProp requires size[0]*size[1] == edges.shape[1]"));

	// outputs
	Tensor *labels_tensor=NULL; // labels output allocation and pointer
	OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{(B*H*W)}, &labels_tensor));
	auto labels_flat = labels_tensor->flat<int>();
	int *labels = &(labels_flat(0));

	Tensor *num_segments_tensor=NULL; // num_segments output allocation and pointer
	OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{B}, &num_segments_tensor));
	auto num_segments_flat = num_segments_tensor->flat<int>();
	int *num_segments = &(num_segments_flat(0));

	// assign the labels via labelprop and compute num_segments per example
	assignLabels(B, H, W, k, num_steps_, defensive_, edges, labels, num_segments);
    }
};

REGISTER_KERNEL_BUILDER(Name("LabelProp").Device(DEVICE_CPU), LabelPropOp);

using namespace std;

// assigns an integer label for each b, h, w in increasing order across examples
static void assignLabels(int B, int H, int W, int k, int num_steps, bool defensive, const bool *edges, int *labels, int *num_segments){
    // constants
    int edges_per_ex = H*W*(std::pow((2*k + 1), 2)); int N = H*W;
    bool relabel=true; bool shuffle=true;

    int segments_now=0;

    // loop across batches
    for (int b=0; b<B; b++){
		// do label prop w random seed
		std::srand(std::time(0));
		PixGraph G(H, W, k, &edges[b*edges_per_ex]); // edges for this example
		for (int n=0; n<num_steps; n++)
		    G.labelPropStep(defensive);
		G.setNumLabels(relabel, shuffle, segments_now);

		// assign outputs
		int offset = b*N;
		for (int i=0; i<N; i++)
		    labels[offset + i] = G.V[i];

		num_segments[b] = G.num_labels;

		// update
		segments_now = segments_now + G.num_labels;
    }
}
