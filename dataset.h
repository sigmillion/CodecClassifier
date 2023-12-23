#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <stdlib.h>

// Structure used for sorting
typedef struct dimdata {
  unsigned char x; // pixel data
  unsigned char y; // label
  int i;   // index
} dimdata;

// Compare two dimdata objects
int comp(const void *a, const void *b) {
    dimdata* a1 = (dimdata*)a;
    dimdata* b1 = (dimdata*)b;
    if ((*a1).x > (*b1).x) return 1;
    else if ((*a1).x < (*b1).x) return -1;
    else return 0;
}

// Main dataset class
class dataset {
 public:
  int num_classes;
  int num_instances;
  int num_dimensions;
  unsigned char* X; // feature data
  unsigned char* y; // labels
  // Don't need splits array as in Matlab code.  Just use integers
  // from 0, 1, ..., 254 = 255 values.  This works because our logic
  // is x[f_i] <= t_i.  That equal allows points with 0 value to go left
  // and others to go right.
  int num_splits;
  dimdata* Xyi;
  
  dataset() {
    num_classes = 10; // 0, 1, ..., 9
    num_instances = 0;
    num_dimensions = 784; // = 28*28
    num_splits = 255; // Split at 0, 1, 2, ..., 254
    X = NULL;
    y = NULL;
    Xyi = NULL;
  }

  dataset(const char* datafile, const char* labelfile, int num_instances_to_load=60000) {
    num_classes = 10; // 0, 1, ..., 9
    num_instances = 0;
    num_dimensions = 784; // = 28*28
    num_splits = 255; // Split at 0, 1, 2, ..., 254
    X = NULL;
    y = NULL;
    Xyi = NULL;
    load_features(datafile, num_instances_to_load);
    load_labels(labelfile, num_instances_to_load);
  }
  
  ~dataset() {
    if(X != NULL) { delete [] X; }
    if(Xyi != NULL) { delete [] Xyi; }
    if(y != NULL) { delete [] y; }
  }

  void sort_dimension(int dim) {
    for(int i=0, j=dim; i<num_instances; i++, j+=num_dimensions) {
      Xyi[i].x = X[j]; // pixel value
      Xyi[i].y = y[i]; // label
      Xyi[i].i = i;    // index
    }
    qsort(Xyi,num_instances,sizeof(dimdata),comp);
  }
  
  void load_features(const char *filename, int num_instances_to_load=60000) {
    FILE *fid = fopen(filename,"rb");
    if(fid == NULL) {
      fprintf(stderr,"ERROR: %s cannot be opened.\n",filename);
      return;
    }
    // Skip over the header
    unsigned char *dump = new unsigned char[16];
    size_t num = fread(dump,sizeof(unsigned char),16,fid);

    // Allocate memory
    num_instances = num_instances_to_load;
    X = new unsigned char[num_instances * num_dimensions];
    Xyi = new dimdata[num_instances];
    
    // Read
    num = fread(X,sizeof(unsigned char),num_instances * num_dimensions,fid);
    if(num < num_instances_to_load) {
      fprintf(stderr,"ERROR: Requested %d, received %lu.\n",
	      num_instances_to_load, num);
    }
    fclose(fid);
  }

  void load_labels(const char *filename, int num_instances_to_load=60000) {
    FILE *fid = fopen(filename,"rb");
    if(fid == NULL) {
      fprintf(stderr,"ERROR: %s cannot be opened.\n",filename);
      return;
    }
    // Skip over the header
    unsigned char *dump = new unsigned char[8];
    size_t num = fread(dump,sizeof(unsigned char),8,fid);

    // Allocate memory
    num_instances = num_instances_to_load;
    y = new unsigned char[num_instances];

    // Read
    num = fread(y,sizeof(unsigned char),num_instances,fid);
    if(num < num_instances_to_load) {
      fprintf(stderr,"ERROR: Requested %d, received %lu.\n",
	      num_instances_to_load, num);
    }
    fclose(fid);
  }

  
};

#endif
