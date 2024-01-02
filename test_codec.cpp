#include "dataset.h"
#include "codec.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
  // Instantiate a dataset and load the data
  dataset DS("/Users/Jake/Downloads/MNIST_ORG/train-images.idx3-ubyte",
	     "/Users/Jake/Downloads/MNIST_ORG/train-labels.idx1-ubyte",60000); // Load decimated
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

#if 1
    C.load("training_60k_028.model");
    C.set_codewords(DS); // Have to do this after loading a new model before training next
                         // because the codewords are needed to extend the existing model
                         // and train the new classifier.  When doing batch processing,

  
  int num_classifiers = 30;
  for(int i=29; i<num_classifiers; i++) {
    C.train_next(DS);

    char filename[100];
    sprintf(filename,"training_60k_%03d.model",i);
    C.save(filename);

    C.compute_test_error(DS_test);
    //C.build_rectangles();
    //C.predict_and_fix(DS_test);
    //C.compute_test_error(DS_test);


    //C.print();
    //C.save("model.dat");
    
    //codec D(DS.num_classes);
    //D.load("model.dat");
    //D.print();
    //D.compute_test_error(DS_test);
        
  } // End loop over classifiers
  //C.save("model.dat");
#endif
#if 0
  C.load("model.dat");
  C.build_rectangles();
  C.testcode();
  //C.enc.pop_back();
  //C.train_next(DS);
  //C.compute_test_error(DS_test);
#endif
#if 0
  if(0) {
    int num_classifiers = 4;
    for(int i=0; i<num_classifiers; i++) {
      C.train_next(DS);
      C.compute_test_error(DS_test);
      char filename[100];
      sprintf(filename,"training_600_%03d.model",i);
      C.save(filename);
      printf("=========================================\n");
      printf("Model = %d\n",i);
      C.print();
    }
  } else {
    C.load("training_600_002.model");
    C.set_codewords(DS); // Have to do this after loading a new model before training next
                         // because the codewords are needed to extend the existing model
                         // and train the new classifier.  When doing batch processing,
    printf("=========================================\n");
    printf("Model = %d\n",2);
    C.print();
    C.compute_train_error(DS);
    C.compute_test_error(DS_test);
    C.train_next(DS);
    C.compute_test_error(DS_test);
    C.print();
  }
#endif
  
  return 0;
}
