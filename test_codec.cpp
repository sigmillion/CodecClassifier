#include "dataset.h"
#include "codec.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
  // Instantiate a dataset and load the data
  dataset DS("/Users/Jake/Downloads/MNIST_ORG/train-images.idx3-ubyte",
	     "/Users/Jake/Downloads/MNIST_ORG/train-labels.idx1-ubyte",6000); // Load decimated
  //dataset DS_test("/Users/Jake/Downloads/MNIST_ORG/train-images.idx3-ubyte",
  //"/Users/Jake/Downloads/MNIST_ORG/train-labels.idx1-ubyte",60000); // Load everything
  //dataset DS("/Users/Jake/Downloads/MNIST_ORG/train-images.idx3-ubyte",
  //"/Users/Jake/Downloads/MNIST_ORG/train-labels.idx1-ubyte",60000); // Load decimated
  //dataset DS("/Users/Jake/Downloads/MNIST_ORG/t10k-images.idx3-ubyte",
  //"/Users/Jake/Downloads/MNIST_ORG/t10k-labels.idx1-ubyte",10000); // Load everything
  dataset DS_test("/Users/Jake/Downloads/MNIST_ORG/t10k-images.idx3-ubyte",
		  "/Users/Jake/Downloads/MNIST_ORG/t10k-labels.idx1-ubyte",10000); // Load everything
  codec C(DS.num_classes); // Instantiate a codec classifier with 30 base classifiers
  //C.train_batch(DS,30);

  // Print dataset sizes
  printf("Training set size = %d\n",DS.num_instances);
  printf("Testing  set size = %d\n",DS_test.num_instances);

  int num_classifiers = 10;
  for(int i=0; i<num_classifiers; i++) {
    C.train_next(DS);
    C.build_rectangles();
    C.predict_and_fix(DS_test);
    C.compute_test_error(DS_test);
    //C.print();
        
  } // End loop over classifiers
  
  return 0;
}
