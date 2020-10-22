#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "graphs.h"

using namespace tensorflow;

// Finds the C largest connected components in each example of a [B,N,N] batch of boolean graph edge matricies
REGISTER_OP("ConnectedComponents")
    .Attr("num_ccs: int")
    .Input("edges: bool")
    .Input("mask: bool")
    .Output("ccids: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* context) {
	    context->set_output(0, context->input(1));
	    return Status::OK();
	});

static void ccsearch(int b, int n, int C, const bool* edges, const bool* mask, int* ccids);

class ConnectedComponentsOp : public OpKernel{
private:
    int num_ccs_;
public:
    explicit ConnectedComponentsOp(OpKernelConstruction* context):OpKernel(context){
	OP_REQUIRES_OK(context, context->GetAttr("num_ccs", &num_ccs_));
    }
    void Compute(OpKernelContext * context) override {

	const Tensor& edges_tensor=context->input(0);
	const Tensor& mask_tensor=context->input(1);
	OP_REQUIRES(context,edges_tensor.dims()==3,errors::InvalidArgument("ConnectedComponents requires edges to be of shape (batch,#points,#points)"));
	int b=edges_tensor.shape().dim_size(0);
	int n=edges_tensor.shape().dim_size(1);
	OP_REQUIRES(context,mask_tensor.dims()==2,errors::InvalidArgument("ConnectedComponents requires mask to be of shape (batch,#points)"));
	OP_REQUIRES(context,mask_tensor.shape().dim_size(0)==b,errors::InvalidArgument("ConnectedComponents expects edges and mask to have same batch size"));
	OP_REQUIRES(context,mask_tensor.shape().dim_size(1)==n,errors::InvalidArgument("ConnectedComponents expects edges and mask to have same number of points"));
	auto edges_flat=edges_tensor.flat<bool>();
	const bool* edges=&edges_flat(0);
	auto mask_flat=mask_tensor.flat<bool>();
	const bool* mask=&mask_flat(0);
	Tensor* ccids_tensor=NULL;
	OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n},&ccids_tensor));
	auto ccids_flat=ccids_tensor->flat<int>();
	int* ccids=&(ccids_flat(0));
	// find up to num_ccs
	ccsearch(b, n, num_ccs_, edges, mask, ccids);
    }
};

REGISTER_KERNEL_BUILDER(Name("ConnectedComponents").Device(DEVICE_CPU), ConnectedComponentsOp);

using namespace std;

// finds the C largest connected components for each n particles in a batch of size b
// returns cc_ids with lower id numbers corresponding to larger connected components
static void ccsearch(int b, int n, int C, const bool* edges, const bool* mask, int* ccids){
    // loop across batches
    for (int i=0; i<b; i++){
	bool visited[n]; // which particles/nodes have been visited

	for (int v=0; v<n; v++){
	    visited[v] = (mask[i*n + v]) ? false : true; // if a particle is fake, never visit
	}

	// construct the graph for this example and add edges
	Graph G(n, false);
	for (int v=0; v<n; v++)
	    // only need to do lower triangle + diagonal of edge mat
	    for (int w=0; w<=v; w++){
		if (edges[i*n*n + v*n + w])
		    G.addEdge(v,w);
	    }

	// compute the connected components
	G.connectedComponents(visited);

	// sort by size, largest first
	int num_ccs = G.cc_sizes.size();
	vector<int> cc_order(num_ccs);
	std::iota(cc_order.begin(), cc_order.end(), 0);
	std::sort(cc_order.begin(), cc_order.begin()+num_ccs,
		  [&G](int i, int j) {return G.cc_sizes[i]>G.cc_sizes[j];});

	// how to map originally-assigned cc_ids to size-ranked cc_ids up to C (max components)
	vector<int> cc_map(num_ccs);
	for (int idx = 0; idx < num_ccs; idx++)
	    cc_map[cc_order[idx]] = idx;

	// assign cc ids only to the largest connected components
	// fake particles and any cc >= C is set to cc_id = C
	for (int v=0; v<n; v++){
	    int v_id = G.cc_ids[v];
	    if (v_id > 0)
		ccids[i*n + v] = (cc_map[v_id] < C) ? cc_map[v_id] : C;
	    else
		ccids[i*n + v] = C;
	}
    }
}

