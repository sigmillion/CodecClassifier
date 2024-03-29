#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <stdlib.h>
#include <bitset>
#include <functional>

#define MAX_NUM_BITS 30
typedef std::bitset<MAX_NUM_BITS> codeword;
typedef std::hash<codeword> hash;

// Structure used for sorting
typedef struct dimdata {
  unsigned char x; // pixel data
  //unsigned char y; // label
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
  codeword* C; // codewords

  void init(void) {
    num_classes = 10; // 0, 1, ..., 9
    num_instances = 0;
    num_dimensions = 784; // = 28*28
    num_splits = 255; // Split at 0, 1, 2, ..., 254
    X = NULL;
    y = NULL;
    Xyi = NULL;
    C = NULL;
  }
  
  dataset() { init(); }

  dataset(dataset & ds, int i) :
    X(ds.X+i*ds.num_dimensions),
    y(ds.y+i),
    C(ds.C+i),
    Xyi(NULL),
    num_classes(ds.num_classes),
      num_instances(1),
    num_dimensions(ds.num_dimensions),
    num_splits(ds.num_splits) {}

  void set(dataset & ds, int i) {
    X = ds.X+i*ds.num_dimensions;
    y = ds.y+i;
    C = ds.C+i;
   }

  dataset(const char* datafile, const char* labelfile, int num_instances_to_load, int num_instances_to_skip=0) {
    init(); // Set up all the initial parameters
    load_features_decimated(datafile, num_instances_to_load, num_instances_to_skip);
    load_labels_decimated(labelfile, num_instances_to_load, num_instances_to_skip);
  }
  
  ~dataset() {
    if(X != NULL) { delete [] X; }
    if(Xyi != NULL) { delete [] Xyi; }
    if(y != NULL) { delete [] y; }
    if(C != NULL) { delete [] C; }
  }

  void sort_dimension(int dim) {
    for(int i=0, j=dim; i<num_instances; i++, j+=num_dimensions) {
      Xyi[i].x = X[j]; // pixel value
      //Xyi[i].y = y[i]; // label
      Xyi[i].i = i;    // index
    }
    qsort(Xyi,num_instances,sizeof(dimdata),comp);
  }
  
  void load_features(const char *filename, int num_instances_to_load) {
    FILE *fid = fopen(filename,"rb");
    if(fid == NULL) {
      fprintf(stderr,"ERROR: %s cannot be opened.\n",filename);
      exit(0);
    }
    // Skip over the header
    unsigned char *dump = new unsigned char[16];
    size_t num = fread(dump,sizeof(unsigned char),16,fid);

    // Allocate memory
    X = new unsigned char[num_instances_to_load * num_dimensions];
    Xyi = new dimdata[num_instances_to_load];
    C = new codeword [num_instances_to_load];
    
    // Read
    num = fread(X,sizeof(unsigned char),num_instances_to_load * num_dimensions,fid);
    if(num < num_instances_to_load * num_dimensions) {
      fprintf(stderr,"ERROR: Requested %d, received %lu.\n",
	      num_instances_to_load, num);
      exit(0);
    }
    num_instances = num_instances_to_load;
    fclose(fid);
  }

  void load_features_decimated(const char *filename, int num_instances_to_load, int num_instances_to_skip) {
    FILE *fid = fopen(filename,"rb");
    if(fid == NULL) {
      fprintf(stderr,"ERROR: %s cannot be opened.\n",filename);
      exit(0);
    }
    // Skip over the header
    unsigned char *dump = new unsigned char[16];
    size_t num = fread(dump,sizeof(unsigned char),16,fid);

    // Allocate memory
    // Set up for loading a decimated data set as is done in Matlab
    // so that I can compare these results to the Matlab results.
    X = new unsigned char[num_instances_to_load * num_dimensions];
    Xyi = new dimdata[num_instances_to_load];
    C = new codeword[num_instances_to_load];
    
    // Read
    num_instances = 0;
    for(int i=0; i<num_instances_to_load; i++) {
      num = fread(X+i*num_dimensions,sizeof(unsigned char),num_dimensions,fid);
      if(num < num_dimensions) {
	fprintf(stderr,"ERROR: Requested %d, received %lu.\n",
		num_dimensions, num);
	exit(0);
      }
      fseek(fid,sizeof(unsigned char)*num_dimensions*num_instances_to_skip,SEEK_CUR);
      num_instances++;
    }
    fclose(fid);
    if(num_instances != num_instances_to_load) {
      fprintf(stderr,"ERROR: Did not load the right number of instances.  Requested %d and received %d.\n",
	      num_instances_to_load, num_instances);
      exit(0);
    }
  }

  void load_labels(const char *filename, int num_instances_to_load) {
    FILE *fid = fopen(filename,"rb");
    if(fid == NULL) {
      fprintf(stderr,"ERROR: %s cannot be opened.\n",filename);
      exit(0);
    }
    // Skip over the header
    unsigned char *dump = new unsigned char[8];
    size_t num = fread(dump,sizeof(unsigned char),8,fid);

    // Allocate memory
    y = new unsigned char[num_instances_to_load];

    // Read
    num = fread(y,sizeof(unsigned char),num_instances_to_load,fid);
    if(num < num_instances) {
      fprintf(stderr,"ERROR: Requested %d, received %lu.\n",
	      num_instances_to_load, num);
      exit(0);
    }
    fclose(fid);
  }

  void load_labels_decimated(const char *filename, int num_instances_to_load, int num_instances_to_skip) {
    FILE *fid = fopen(filename,"rb");
    if(fid == NULL) {
      fprintf(stderr,"ERROR: %s cannot be opened.\n",filename);
      exit(0);
    }
    // Skip over the header
    unsigned char *dump = new unsigned char[8];
    size_t num = fread(dump,sizeof(unsigned char),8,fid);

    // Allocate memory
    y = new unsigned char[num_instances_to_load];

    // Read
    for(int i=0; i<num_instances; i++) {
      num = fread(y+i,sizeof(unsigned char),1,fid);
      if(num < 1) {
	fprintf(stderr,"ERROR: Requested %d, received %lu.\n",
		num_instances_to_load, num);
	exit(0);
      }
      fseek(fid,sizeof(unsigned char)*num_instances_to_skip,SEEK_CUR);
    }
    fclose(fid);
  }
  
};

#endif
