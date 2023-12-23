#include "dataset.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
  // Instantiate a dataset and load the data
  dataset ds("/Users/Jake/Downloads/MNIST_ORG/train-images.idx3-ubyte",
	     "/Users/Jake/Downloads/MNIST_ORG/train-labels.idx1-ubyte",20);

  /// Sort on a dimension
  ds.sort_dimension(181);

  // Print out the sorted data
  for(int i=0; i<20; i++) {
    printf("%6d, %3d, %1d, %6d\n",i,ds.Xyi[i].x,ds.Xyi[i].y,ds.Xyi[i].i);
  }
  
  return 0;
}
