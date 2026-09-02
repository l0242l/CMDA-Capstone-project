// uh i dont dont how to read partquet file in c so i will have to conver them into csv first using python


// Pallaraztion options
//1. Cuda C fastest, hardest
//2. mpi need multiple computers, not fesable on my pc.
//3. omp, uses cpu threads, not as fast, easy 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <omp.h>

double dtw_distance(double* u, double* v, int dim) {
    // We use a Variable Length Array (VLA) for the DTW matrix.
    // This is much faster than malloc/free inside a function called millions of times.
    // This assumes 'dim' is not so large that it overflows the stack.
    double dtw_matrix[dim][dim];
    
    double cost;

    // Initialize the first cell
    cost = (u[0] - v[0]) * (u[0] - v[0]);
    dtw_matrix[0][0] = cost;

    // Initialize the first row
    for (int j = 1; j < dim; j++) {
        cost = (u[0] - v[j]) * (u[0] - v[j]);
        dtw_matrix[0][j] = cost + dtw_matrix[0][j - 1];
    }

    // Initialize the first column
    for (int i = 1; i < dim; i++) {
        cost = (u[i] - v[0]) * (u[i] - v[0]);
        dtw_matrix[i][0] = cost + dtw_matrix[i - 1][0];
    }

    // Fill the rest of the matrix
    for (int i = 1; i < dim; i++) {
        for (int j = 1; j < dim; j++) {
            cost = (u[i] - v[j]) * (u[i] - v[j]);
            // Find the minimum of the three neighbors
            double min_prev = fmin(dtw_matrix[i - 1][j - 1], 
                                 fmin(dtw_matrix[i - 1][j], 
                                      dtw_matrix[i][j - 1]));
            dtw_matrix[i][j] = cost + min_prev;
        }
    }

    // The final distance is in the bottom-right corner
    return dtw_matrix[dim - 1][dim - 1];
}


// calculate the distance squared between dim dimensional vectors u and v
double vec_dist_sq (double* u, double* v, int dim) {
    double dist_sq = 0;
    for (int i=0;i<dim;i++) {
        dist_sq += (u[i]-v[i])*(u[i]-v[i]);
    }
    return dist_sq;
}

// read len vectors in dim dimensional space from stdin into data array
void vec_read_dataset (double* data, int len, int dim) {
    for (int i=0;i<len;i++) {
        for (int j=0;j<dim;j++) {
            if (scanf("%lf",&(data[i*dim+j])) != 1) {
                printf ("error reading dataset\n");
                exit(1);
            }
        }
    }
}

// v = 0
void vec_zero (double* v, int dim) {
    for (int i=0;i<dim;i++) {
        v[i] = 0;
    }
}

// w = u + v
void vec_add (double* u, double* v, double* w, int dim) {
    for (int i=0;i<dim;i++) {
        w[i] = u[i] + v[i];
    }
}

// w = cv
void vec_scalar_mult (double* v, double c, double* w, int dim) {
    for (int i=0;i<dim;i++) {
        w[i] = c*v[i];
    }
}

// performs the copy v->data[i] = w->data[i] for all i
void vec_copy (double* v, double* w, int dim) {
    for (int i=0;i<dim;i++) {
        v[i] = w[i];
    }
}

// calculate the arg max for farthest first
int calc_arg_max (double* data, int len, int dim, int* centers, int m) {
    int arg_max;
    double cost_sq = 0;
    
    // Use parallel for if len is large, but be careful with shared variables
    // Simple loop for clarity:
    for (int i=0;i<len;i++) {
        double min_dist_sq = DBL_MAX;
        for (int j=0;j<m;j++) {
            // *** CHANGE: Using dtw_distance instead of vec_dist_sq ***
            double dist_sq = dtw_distance(data+i*dim,data+centers[j]*dim,dim);
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
            }
        }
        if (min_dist_sq > cost_sq) {
            cost_sq = min_dist_sq;
            arg_max = i;
        }
    }
    return arg_max;
}

// for every point find the cluster index (i.e. the index of the closest mean)
// clusters is an output array of size len that contains the cluster index of each point
void find_clusters (double* data, int len, int dim, double* kmeans, int k, int* clusters) {

    // This loop is highly parallelizable
    #pragma omp parallel for
    for (int i = 0; i<len; i++){
        double min_dist_sq = DBL_MAX;
        int best_cluster = 0;
        for (int j = 0; j<k; j++){
            // *** CHANGE: Using dtw_distance instead of vec_dist_sq ***
            double dist_sq = dtw_distance(data+i*dim,kmeans+j*dim,dim);
            if (dist_sq < min_dist_sq){
                min_dist_sq = dist_sq;
                best_cluster = j;
            }
        }
        clusters[i] = best_cluster;
    }
    
}

// calculate the next kmeans
// clusters is an input array of size len that contains the cluster index of each point
// kmeans_next is an output array of size k*dim that contains the next kmeans
// do not assume that the kmeans_next array has been initialized
void calc_kmeans_next (double* data, int len, int dim, double* kmeans_next, int k, int* clusters) {

    for (int i=0;i<k;i++) {
        for (int j=0;j<dim;j++) {
            kmeans_next[i*dim+j] = 0;
        }
    }

    int* counts = (int*)calloc(k, sizeof(int));
    
    // This part is a reduction and can be tricky to parallelize
    // A simple loop is safer unless performance is critical
    for (int i=0;i<len;i++) {
        int cluster = clusters[i];
        counts[cluster]++;
        for (int j=0;j<dim;j++) {
            kmeans_next[cluster*dim+j] += data[i*dim+j];
        }
    }

    #pragma omp parallel for
    for (int i=0;i<k;i++) {
        if (counts[i] > 0){
            for (int j=0;j<dim;j++) {
                kmeans_next[i*dim+j] /= counts[i];
            }
        }
    }


    free(counts);
    
}

int main (int argc, char** argv) {

    // get k, m, and num_threads from command line
    if (argc < 4) {
        printf ("Command usage : %s k m num_threads\n",argv[0]);
        return 1;
    }
    int k = atoi(argv[1]);
    int m = atoi(argv[2]);
    int num_threads = atoi(argv[3]);
    omp_set_num_threads(num_threads);

    // read the number of points and the dimension of each point    
    int len, dim;
    if (scanf("%d %d",&len,&dim) != 2) {
        printf ("error reading the length and dimension of the dataset\n");
        return 1;
    }

    // allocate the data array on the heap using malloc
    double* data = (double*)malloc(len*dim*sizeof(double));
    if (data == NULL) {
        printf ("malloc failed to allocate data array\n");
        return 1;
    }

    // read the dataset from stdin
    vec_read_dataset(data,len,dim);

    // allocate the clusters array
    int* clusters = (int*)malloc(len*sizeof(int));

    // start the timer
    double start_time, end_time;
    start_time = omp_get_wtime();
    
    // find k centers using the farthest first algorithm
    int centers[k];
    centers[0] = 0;
    for (int m=1;m<k;m++) {
        centers[m] = calc_arg_max(data,len,dim,centers,m);
    }

    // initialize kmeans using the k centers from farthest first
    double kmeans[k*dim];
    for (int i=0;i<k;i++) {
        vec_copy(kmeans+i*dim,data+centers[i]*dim,dim);
    }

    // update kmeans m times
    double kmeans_next[k*dim];
    for (int i=0;i<m;i++) {
    find_clusters(data,len,dim,kmeans,k,clusters);
        calc_kmeans_next(data,len,dim,kmeans_next,k,clusters);
        vec_copy(kmeans,kmeans_next,k*dim);
    }

    // stop the timer
    end_time = omp_get_wtime();

#ifdef TIMING
    printf ("(%d,%.4f),",num_threads,(end_time-start_time));
#else
    // print out the number of threads
    printf ("# num_threads = %d\n",num_threads);
    
    // print out wall time used
    printf ("# wall time used = %.4f sec\n",end_time-start_time);

    // print the results
    for (int i=0;i<k;i++) {
        for (int j=0;j<dim;j++) {
            printf ("%.5lf ",kmeans[i*dim+j]);
        }
        printf ("\n");
    }
#endif
    
    // free the dynamically allocated memory
    free (data);
    free (clusters);
}