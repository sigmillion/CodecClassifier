#include "dataset.h"
#include "codec.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
  // Instantiate a dataset and load the data
  dataset DS("/Users/Jake/Downloads/MNIST_ORG/train-images.idx3-ubyte",
	     "/Users/Jake/Downloads/MNIST_ORG/train-labels.idx1-ubyte",6000);
  codec C(DS.num_classes); // Instantiate a codec classifier with 30 base classifiers
  C.train_batch(DS,30);
  return 0;
}
