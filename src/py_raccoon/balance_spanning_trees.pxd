

cdef packed struct Edge:
    int a
    int b
    signed char weight

cdef packed struct LcaResult:
    int a
    int b
    int lca
    signed char bal

cdef packed struct Graph_c:
    int** neighbors
    signed char** weights

cdef LcaResult* lowest_common_ancestor(int[:] parent, signed char[:] parent_weight, Edge[:] node_pairs);

cdef void calc_property_fast(int[:] parent, double[:] result, double root_val, int[:] degree, double (*update_fun)(int, int, double, int[:]));

cdef int uniform_spanning_tree_c(int size, int[:] degree, int** neighbors, signed char** weights, int[:] parent, signed char[:] parent_weight, rnd);

cdef Graph_c to_Graph_c(int size, int[:] degree, G);

cdef void free_neighbors_and_weights(int size, int** neighbors, signed char** weights);

cdef int** graph_to_neighbors(int size, int[:] degree, G);

cdef void free_graph_neighbors(int size, int** neighbors);