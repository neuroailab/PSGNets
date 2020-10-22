#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "graphs.h"

using namespace tensorflow;

// Propagates labels on graphs with arbitrary connectivity. Operates on
// B graphs with num_nodes[b] nodes each, with edges defined by the
// variable length tensor edges.
// Returns the labels of each node in each example.

REGISTER_OP("LabelPropFc")
    .Attr("num_steps: int") // num steps to run op
    .Attr("sort_edges: bool = true") // whether to sort the edges in increasing order of example
    .Input("num_nodes: int32") // [B] number of nodes in each of B examples
    .Input("edges: int32") // edges across all examples, [?,3] where edges[i] = (batch_ind, sender_node_idx, receiver_node_idx)
    .Output("labels: int32") // [sum(num_nodes)] of new labels for each node
    .Output("num_segments: int32") // [B] number of segments (unique labels) per example
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->UnknownDim()});
	c->set_output(0, output);
	c->set_output(1, c->Vector(c->Dim(c->input(0),0)));
	return Status::OK();
	});

static void assignLabelsFC(int B, int E, const int *num_nodes, const int *edges, int *labels, int *num_segments, int num_steps, bool sort_edges); // declaration

class LabelPropFcOp : public OpKernel{
private:
    int num_steps_;
    bool sort_edges_;
public:
    explicit LabelPropFcOp(OpKernelConstruction *context):OpKernel(context){
	OP_REQUIRES_OK(context, context->GetAttr("num_steps", &num_steps_));
	OP_REQUIRES_OK(context, context->GetAttr("sort_edges", &sort_edges_));
    }
    void Compute(OpKernelContext *context) override {
	// num_nodes input
	const Tensor &num_nodes_tensor = context->input(0);
	OP_REQUIRES(context, num_nodes_tensor.dims()==1, errors::InvalidArgument("num_nodes must be a rank 1 tensor indicating number of nodes per example."));
	int B = num_nodes_tensor.shape().dim_size(0); // num examples
	auto num_nodes_flat = num_nodes_tensor.flat<int>();
	const int *num_nodes = &num_nodes_flat(0); // input pointer to num_nodes

	// edges input
	const Tensor &edges_tensor = context->input(1);
	OP_REQUIRES(context, edges_tensor.dims()==2, errors::InvalidArgument("edges must be a rank 2 tensor where the first dimension indexes the edges."));
	OP_REQUIRES(context, edges_tensor.shape().dim_size(1)==3, errors::InvalidArgument("edges.shape[1] must be 3 where each edge is (example_idx, Lnode_idx, Rnode_idx"));
	int E = edges_tensor.shape().dim_size(0);
	auto edges_flat = edges_tensor.flat<int>();
	const int *edges = &edges_flat(0);

	// compute number of total nodes
	int total_nodes = 0;
	for (int b=0; b<B; b++)
	    total_nodes = total_nodes + num_nodes[b];

	// labels output
	Tensor *labels_tensor = NULL;
	OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{total_nodes}, &labels_tensor));
	auto labels_flat = labels_tensor->flat<int>();
	int *labels = &labels_flat(0); // pointer to labels output

	// num_segments output
	Tensor *num_segments_tensor = NULL;
	OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{B}, &num_segments_tensor));
	auto num_segments_flat = num_segments_tensor->flat<int>();
	int *num_segments = &num_segments_flat(0); // pointer to num_segments output

	// assign labels
	assignLabelsFC(B, E, num_nodes, edges, labels, num_segments, num_steps_, sort_edges_);
    }
};

REGISTER_KERNEL_BUILDER(Name("LabelPropFc").Device(DEVICE_CPU), LabelPropFcOp);

using namespace std;

// constructs a graph of size num_nodes[b] for b in range(B), adds the appropriate edges, and does labelprop to reassign node labels.
// if sort_edges, then the edges
static void assignLabelsFC(int B, int E, const int *num_nodes, const int *edges, int *labels, int *num_segments, int num_steps, bool sort_edges){
    // either make a copy of edges and sort or just reference
    int **sedges = new int*[E];
    for (int e=0; e<E; e++){
    	sedges[e] = new int[3];
    	for (int i=0; i<3; i++)
    	    sedges[e][i] = edges[e*3 + i];
    }
    if (sort_edges)
    	std::sort(&sedges[0], &sedges[E], [&](int *ei, int *ej){return ei[0] < ej[0];});

    // debug print
    // printf("Edges post sorting\n");
    // for (int e=0; e<E; e++){
    // 	for (int i=0; i<3; i++)
    // 	    std::cout << sedges[e][i] << " ";
    // 	std::cout << endl;
    // }
    // std::cout << endl;

    // set labels and num segments output
    int offset = 0; int edge_ctr = 0;
    int nodes_so_far = 0;
    for (int b=0; b<B; b++){

    	int V = num_nodes[b]; // num nodes in this graph
	Graph G(V, true); // init graph for labelprop

	// add edges
	while ((E - edge_ctr) && (sedges[edge_ctr][0] == b)){
	    // std::cout << sedges[edge_ctr][0] << " " << sedges[edge_ctr][1] << " " << sedges[edge_ctr][2] << endl;
	    G.addEdge(sedges[edge_ctr][1], sedges[edge_ctr][2]);
	    edge_ctr++;
	}

	// do the labelprop and reset labels
	std::srand(std::time(0));
	for (int n=0; n<num_steps; n++)
	    G.labelPropStep(false);
	G.setNumLabels(true, offset);

	// assign outputs
	for (int v=0; v<V; v++)
	    labels[nodes_so_far + v] = G.cc_ids[v];
	num_segments[b] = G.num_labels;

	// update
	offset = offset + G.num_labels;
	nodes_so_far = nodes_so_far + V;
    }

    // free memory
    for (int e=0; e<E; e++)
    	delete[] sedges[e];
    delete[] sedges;

}
