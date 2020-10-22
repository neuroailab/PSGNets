#include "graphs.h"
#include "math.h"
#include <algorithm>
#include <bits/stdc++.h>
#include <chrono>

using namespace std;

int mostCommonElement(std::vector<int> vect){

    // sort vector
    std::sort(vect.begin(), vect.end());

    int mode = -1; int max_count = 0; int count = 0;
    int current = *vect.begin();
    for (std::vector<int>::iterator w=vect.begin(); w !=vect.end(); ++w){
	if (*w == current){
	    count++;
	    if (count > max_count){
		max_count = count;
		mode = *w;
	    }
	}
	else
	    count = 0;
	current = *w;
    }
    return mode;
}

Graph::Graph(int v, bool labelprop=false)
    : num_labels{v}, labelprop{labelprop}
{
    this->V = v;
    this->adj = new std::vector<int>[v];
    this->cc_ids = new int[v];
    // initialization for labelprop
    if (labelprop){
	// init labels and self-edges
	for (int i=0; i<V; i++){
	    cc_ids[i] = i;
	    adj[i].push_back(i);
	}
	// init order
	this->order = new int[V];
	for (int i=0; i<V; i++)
	    order[i] = i;

    }
    else{
	for (int i=0; i<V; i++)
	    cc_ids[i] = 0;
	int cc_size_counter = 0;
	std::vector<int> cc_sizes;
    }

}

Graph::~Graph(){
    delete[] adj;
    delete[] cc_ids;
    if (labelprop)
	delete[] order;
}

void Graph::addEdge(int v, int w){
    adj[v].push_back(w);
    adj[w].push_back(v);
}

void Graph::connectedComponents(bool visited[]){

    // use DFS to find and label all the ccs
    cc_sizes.push_back(0);
    int cc = 1;
    for (int v=0; v<V; v++){
	if (!visited[v]){
	    cc_size_counter = 1;
	    DFSUtil(v, visited, cc);
	    cc_sizes.push_back(cc_size_counter);
	    cc++;
	}
    }
}

void Graph::DFSUtil(int v, bool visited[], int cc){
    visited[v] = true;
    cc_ids[v] = cc;
    std::vector<int> connected_to_v = adj[v];
    std::vector<int>::iterator w;
    for (w = connected_to_v.begin(); w != connected_to_v.end(); ++w){
	if (!visited[*w]){
	    cc_size_counter++;
	    DFSUtil(*w, visited, cc);
	}
    }
}

void Graph::labelPropStep(bool defensive=false){
    // shuffle the order
    std::random_shuffle(&order[0], &order[V]);

    // a step of asynchronous label propagation
    int v; std::vector<int> v_edges;
    for (int i=0; i<V; i++){
	v = order[i];
	v_edges = adj[v]; // vector of connected nodes
	std::vector<int> labels_connected_to_v;
	std::vector<int>::iterator it;
	for (it=v_edges.begin(); it!=v_edges.end(); ++it)
	    labels_connected_to_v.push_back(cc_ids[*it]);

	// update label
	cc_ids[v] = mostCommonElement(labels_connected_to_v);
    }
}

void Graph::setNumLabels(bool relabel=true,
			 int offset=0){
    // set number of unique labels
    // optionally add an offset to each label value
    std::vector<int> labels(cc_ids, cc_ids+V);
    std::sort(labels.begin(), labels.end());
    int lmap[V];
    int num_unique = 0; int elem = -1;
    for (std::vector<int>::iterator it=labels.begin(); it!=labels.end(); ++it){
	if (*it != elem){
	    lmap[*it] = num_unique; // the new label
	    num_unique++;
	}
	elem = *it;
    }
    num_labels = num_unique;

    if (relabel){
	for (int i=0; i<V; i++)
	    cc_ids[i] = lmap[cc_ids[i]] + offset;
    }
}

PixGraph::PixGraph(int H, int W, int k, const bool *A)
    : H{H}, W{W}, N{(H*W)}, k{k}, ksize{(2*k+1)}, max_edges{(int(pow(2*k+1, 2)))}
    , step_counter{0}, num_labels(this->N)
{
    this->V = new int[N];
    this->order = new int[N];
    this->degrees = new int[N];
    for (int i=0; i<N; i++){
	V[i] = i; // initialize labels to be unique
	order[i] = i; // initial order (gets shuffled)
	degrees[i] = 0; // before adding edges, all nodes have degree 0
    }
    // initialize edges from A; always have self edge
    this->adj = new std::vector<int>[N];
    for (int v=0; v<N; v++){
	int degree = 0;
	for (int e=0; e<max_edges; e++){
	    if (A[v*max_edges + e]){
		addEdge(v,e);
		degree++;
	    }
	    else if (e == (max_edges-1)/2){
		adj[v].push_back(v); // self edge
		degree++;
	    }
	}
	degrees[v] = degree;
    }

    // initialize order

    // for (int i=0; i<N; i++)
    // 	order[i] = i; // initial order
}

PixGraph::~PixGraph(){
    delete[] V;
    delete[] adj;
    delete[] order;
    delete[] degrees;
}

void PixGraph::addEdge(int v, int e){
    int row = v / W; int col = v % W;
    int h = (e / ksize) - k; int w = (e % ksize) - k;
    bool in_view = ((row + h >= 0) && (row + h < H) &&
		    (col + w >= 0) && (col + w < W));
    if (in_view){
	int ind = (row + h)*W + (col + w);
	adj[v].push_back(ind);
    }
}

void PixGraph::labelPropStep(bool defensive=false){
    // shuffle the order
    std::random_shuffle(&order[0], &order[N]);

    // a step of asynchronous label propagation
    int v; std::vector<int> v_edges;
    for (int i=0; i<N; i++){
	v = order[i];
	v_edges = adj[v]; // vector of inds to connected nodes
	std::vector<int> labels_connected_to_v;
	std::vector<int>::iterator it;
	for (it=v_edges.begin(); it!=v_edges.end(); ++it){
	    if (defensive){
		int e = V[*it];
		for (int j=0; j<degrees[e]; j++)
		    labels_connected_to_v.push_back(e);
	    }
	    else
		labels_connected_to_v.push_back(V[*it]);
	}
	V[v] = mostCommonElement(labels_connected_to_v);
    }

    // updates
    step_counter++;
}

void PixGraph::setNumLabels(bool relabel=false,
			    bool shuffle=false,
			    int offset=0){
    // set the number of unique labels
    // optionally relabel and shuffle the order so 0 isn't in top left
    std::vector<int> labels(V, V+N);
    std::sort(labels.begin(), labels.end());
    int lmap[N];
    int num_unique = 0; int elem = -1;
    for (std::vector<int>::iterator it=labels.begin(); it!=labels.end(); ++it){
	if (*it != elem){
	    lmap[*it] = num_unique; // the new label hash
	    num_unique++;
	}
	elem = *it;
    }
    num_labels = num_unique;

    if (relabel){
	if (shuffle){
	    int new_order[num_labels];
	    for (int j=0; j<num_labels; j++)
		new_order[j] = j + offset;
	    std::random_shuffle(&new_order[0], &new_order[num_labels]);
	    for (int k=0; k<N; k++)
		V[k] = new_order[lmap[V[k]]];
	}
	else{
	    for (int i=0; i<N; i++)
		V[i] = lmap[V[i]] + offset;
	}
    }
}

void PixGraph::printLabels(int width=5){
    std::cout << "number of steps taken: " << step_counter << std::endl;
    std::cout << "number of unique labels: " << num_labels << std::endl;

    for (int i=0; i<H; i++){
	for (int j=0; j<W; j++){
	    std::cout << std::setw (width) << V[i*W + j];
	}
	printf("\n");
    }
    printf("\n");
}

int main(){

    // Test of CC Graphs
    // int N = 5;
    // Graph G(N, false);
    // G.addEdge(1, 0);
    // G.addEdge(2, 3);
    // G.addEdge(3, 4);

    // bool mask[N] = {true, true, true, false, true};
    // bool visited[N];
    // for (int v=0; v<N; v++)
    // 	visited[v] = (mask[v]) ? false : true;

    // G.connectedComponents(visited);

    // printf("ConnComp labels\n");
    // for (int v=0; v<N; v++){
    // 	printf("%d ", G.cc_ids[v]);
    // }
    // std::cout << endl;

    // // Test of Labelprop on a generic Graph
    // std::srand(std::time(0));
    // int M = 100;
    // Graph H(M, true);

    // // add some edges
    // for (int i=0; i<M; i++){
    // 	for (int j=0; j<i; j++){
    // 	    if ((i-j) < 2)
    // 		H.addEdge(i,j);
    // 	}
    // }

    // do the labelprop
    // int num_steps = 10;
    // for (int n=0; n<num_steps; n++)
    // 	H.labelPropStep();
    // H.setNumLabels(false, 0);
    // printf("LabelProp labels\n");
    // for (int v=0; v<M; v++){
    // 	printf("%d ", H.cc_ids[v]);
    // }
    // std::cout << endl;

    // Test of sorting edges
    // int E = 5;
    // int edge_vals[][3] = {
    // 	{1, 0, 2},
    // 	{0, 1, 4},
    // 	{1, 0, 1},
    // 	{2, 1, 3},
    // 	{0, 3, 5}
    // };
    // // int (*edges)[3] = new int[E][3];
    // int **edges = new int*[E];
    // for (int e=0; e<E; e++){
    // 	edges[e] = new int[3];
    // 	for (int i=0; i<3; i++)
    // 	    edges[e][i] = edge_vals[e][i];
    // }
    // printf("Edges pre sorting\n");
    // for (int e=0; e<E; e++){
    // 	for (int i=0; i<3; i++)
    // 	    std::cout << edges[e][i] << " ";
    // 	std::cout << endl;
    // }
    // std::cout << endl;

    // printf("Sorting\n");
    // std::sort(&edges[0], &edges[E], [&](int *ei, int *ej){return ei[0] < ej[0];});
    // printf("Edges post sorting\n");
    // for (int e=0; e<E; e++){
    // 	for (int i=0; i<3; i++)
    // 	    std::cout << edges[e][i] << " ";
    // 	delete[] edges[e];
    // 	std::cout << endl;
    // }

    // delete[] edges;

    // Test of most CommonElement
    // std::vector<int> myvect = {5,2,0,2,2,5,3,3,5,4,2};
    // for (std::vector<int>::iterator it=myvect.begin(); it!=myvect.end(); ++it)
    // 	printf("%d ", *it);
    // printf("\n");
    // printf("\nmost common label: %d\n", mostCommonElement(myvect));

    // Test of Pix Graphs
    int H = 16;
    int W = 32;
    int k = 1;
    int max_edges = pow(2*(k+1), 2);
    bool A[H*W*max_edges];
    std::fill_n(A, H*W*max_edges, true);
    int num_steps = 5;

    std::srand(std::time(0));

    auto start = chrono::steady_clock::now();
    PixGraph G(H, W, k, A);
    for (int n=0; n<num_steps; n++)
    	G.labelPropStep(true);
    auto end = chrono::steady_clock::now();
    G.setNumLabels(true, true, 5); // relabel, shuffle, offset
    G.printLabels();
    std::cout << "time elapsed: " << chrono::duration_cast<chrono::milliseconds>(end-start).count() << endl;

    return 0;
}
