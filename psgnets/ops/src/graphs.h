#include <list>
#include <vector>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <iomanip>

class Graph
{
    int V;
    std::vector<int> *adj; // edge matrix
    int cc_size_counter;
    bool labelprop; // indicator for whether to init for labelprop
    int *order; // labelprop step order

    // to find connected components
    void DFSUtil(int v, bool visited[], int cc);
public:
    Graph(int v, bool labelprop);
    ~Graph();

    int *cc_ids;
    int num_labels;
    std::vector<int> cc_sizes;
    void addEdge(int v, int w);
    void connectedComponents(bool visited[]);
    void labelPropStep(bool defensive); // propagates labels
    void setNumLabels(bool relabel, int offset); // set num labels and relabel
};

class PixGraph
{
    int H; int W;
    std::vector<int> *adj; // adjacency
    int *degrees; // degree of each vertex
    int N; // number of nodes
    int k; int ksize; // radius and half width - 1 of local edge kernel
    int max_edges;
    int step_counter;
    int *order; // order to alter labels

    void addEdge(int v, int e);

public:
    PixGraph(int H, int W, int k, const bool *A);
    ~PixGraph();

    int *V; // labels
    int num_labels; // current number of labels

    void setNumLabels(bool relabel, bool shuffle, int offset); // counts the number of unique labels and reorders
    void labelPropStep(bool defensive); // propagates labels
    void printLabels(int width); // to pretty-print labels
};
