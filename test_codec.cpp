#include "dataset.h"
#include "codec.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
  // Instantiate a dataset and load the data
  dataset DS("/Users/Jake/Downloads/MNIST_ORG/train-images.idx3-ubyte",
	     "/Users/Jake/Downloads/MNIST_ORG/train-labels.idx1-ubyte",6000);
  codec C(30); // Instantiate a codec classifier with 30 base classifiers
  C.train(DS);
  return 0;
}
