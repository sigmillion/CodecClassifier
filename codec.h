#ifndef CODEC_H
#define CODEC_H

#include <iostream>
#include <unordered_map>
#include <bitset>
#include <vector>
#include <string>
#include <cmath>
#include "dataset.h"

#define NUM_CLASSIFIERS 30

// Class for elements of decoder dictionary
class rect {
 public:
  std::vector<int> prob; // number of points from each class
  int num; // total number of points
  unsigned char label; // class label

  rect() {}
  rect(int num) : prob(num, 0) {}
  ~rect() {}
  void reset(void) { std::fill(prob.begin(), prob.end(), 0); }
};

typedef std::bitset<NUM_CLASSIFIERS> codeword;
typedef std::hash<codeword> hash;

class codec {
 public:
  // Encoder parameters
  int num_classifiers;
  unsigned char* t; // thresholds 0, 1, 2, ..., 254
  int *f; // feature indexes
  float *e; // error rate as encoder is grown
  
  // Decoder parameters
  std::unordered_map<codeword, rect, hash> dec;

  codec() {
    t = NULL;
    f = NULL;
    e = NULL;
  }
  codec(int num) {
    if(num != NUM_CLASSIFIERS) {
      fprintf(stderr,"Wrong number of classifiers %d\n",num);
      exit(0);
    }
    num_classifiers = num;
    t = new unsigned char [num];
    f = new int [num];
    e = new float [num];
  }
  ~codec() {
    if(t != NULL) { delete [] t; }
    if(f != NULL) { delete [] f; }
    if(e != NULL) { delete [] e; }
  }

  void train(dataset & ds) {
    codeword c;

    // Create empty dictionaries
    std::vector<int> Mj(ds.num_classes); // Use an array
    std::unordered_map<codeword,int,hash> Nc; // Use a dictionary
    std::vector<std::unordered_map<codeword,int,hash>> Ncj; // Use an array of dictionaries
    for(int j=0; j<ds.num_classes; j++) {
      std::unordered_map<codeword,int,hash> unmap;
      Ncj.push_back(unmap);
    }

    if(0) {
    // Test code
    codeword c0{28};
    codeword c1{2397};
    codeword c2{234};
    Nc.insert({c0,25});
    Nc.insert({c1,81});
    Nc[c2] = 23487;
    printf("%lu\n",Nc.size());
    exit(0);
    }
    
    // Loop over classifiers
    for(int i=0; i<num_classifiers; i++) {
      printf("Working on classifier %3d.\n",i);
      double bestmax = -1e6;
      int bestind = 0;
      unsigned char bestthresh = 0;

      // Loop over dimensions
      for(int d=0; d<ds.num_dimensions; d++) {
      //for(int d=351; d<352; d++) {

	// Initialize all the array and dictionary data structures
	Nc.clear(); // Clear the dictionary
        for(int j=0; j<ds.num_classes; j++) { // One dictionary for each class label
	  Mj[j] = 0; // Initialize array elements to zero
	  Ncj[j].clear(); // Clear the dictionaries
	}

	// Initialize the dictionary counts
	for(int s=0, xs=0; s<ds.num_instances; s++, xs+=ds.num_dimensions) {
	  // Get codeword index for each data point
	  c.reset();
	  for(int j=0; j<i; j++) { if(ds.X[xs+f[j]] > t[j]) { c[j] = 1; } }
	  
	  // Append a 1 to the codeword
	  c[i] = 1;
	  unsigned char y = ds.y[s];
	  // Index the dictionaries and increment counts
	  auto it = Nc.find(c);
	  if(it == Nc.end()) {
	    int tmp = 1;
	    Nc.insert({c,tmp});
	  } else {
	    (it->second)++;
	  }
	  Mj[y]++;
	  it = Ncj[y].find(c);
	  if(it == Ncj[y].end()) {
	    int tmp = 1;
	    Ncj[y].insert({c,tmp});
	  } else {
	    (it->second)++;
	  }
	} // End loop over instances (initialization of dictionary counts)
	
	// Sort current dimension
	ds.sort_dimension(d);

	// Sweep over all the split points
        double themax = -1e6;
        int maxind = 0;
	unsigned char thethresh = 0;
	std::vector<double> MI(ds.num_splits,0.0); // Mutual information array
        int s1 = 0;
	for(int s0=0; s0<ds.num_splits; s0++) {
	  while(s1 < ds.num_instances && ds.X[ds.Xyi[s1].i*ds.num_dimensions + d] <= s0) {
	    // 1. Get codeword for x(s)
	    c.reset();
	    for(int j=0; j<i; j++) { if(ds.X[ds.Xyi[s1].i*ds.num_dimensions + f[j]] > t[j]) { c[j] =1; } }

	    // 2. Append a 1 to the codeword
	    c[i] = 1;
	    unsigned char y = ds.y[ds.Xyi[s1].i];

	    // 3. Index the dictionaries and decrement the counts
	    if(--Nc[c] == 0) { Nc.erase(c); } // Erase dictionary element
	    Mj[y]--; // Decrement array element
	    if(--Ncj[y][c] == 0) { Ncj[y].erase(c); } // Erase dictionary element

	    // 4. Append a 0 to the codeword
	    c[i] = 0;
            
	    // 5. Index the dictionaries and increment the counts
	    auto it = Nc.find(c);
	    if(it == Nc.end()) {
	      int tmp = 1;
	      Nc.insert({c,tmp});
	    } else {
	      (it->second)++;
	    }
	    Mj[y]++;
	    it = Ncj[y].find(c);
	    if(it == Ncj[y].end()) {
	      int tmp = 1;
	      Ncj[y].insert({c,tmp});
	    } else {
	      (it->second)++;
	    }
            //Nc[c]++;
	    //Mj[y]++;
	    //Ncj[y][c]++;

	    s1++;
	  } // Loop over same values

	  if(0) {
	  // Analyze the dictionaries
	  for(int k=0; k<ds.num_classes; k++) {
	    printf("Mj[%2d] = %d\n",k,Mj[k]);
	  }
	  for(auto k: Nc) {
	    std::cout << k.first << " | " << k.second << '\n';
	  }
	  for(int k=0; k<ds.num_classes; k++) {
	    for(auto kk: Ncj[k]) {
	      std::cout << s0 << " " << k << " = " << kk.first << " | " << kk.second << '\n';
	    }
	  }
	  }
	  
          // 6. Compute mutual information
	  for(int j=0; j<ds.num_classes; j++) {
	    int Mjval = Mj[j];
	    for(auto it = Ncj[j].begin(); it != Ncj[j].end(); it++) {
	      c = it->first;
	      int Ncjval = it->second;
	      int Ncval = Nc[c];
	      double wc = ((double)Ncval)/ds.num_instances;
	      double wj = ((double)Mjval)/ds.num_instances;
	      double wcj = ((double)Ncjval)/ds.num_instances;
	      double v = wcj / (wc*wj);
	      double g = v*log(v);
	      if(g>1000.0) { // Clip
		g = 1000.0; fprintf(stderr,"CLIPPED\n");
	      }
	      MI[s0] = MI[s0] + wc*wj*g;
	    } // Loop over dictionary
	  } // Loop over dictionary array
	  //printf("d = %3d, MI[%3d] = %f\n",d,s0,MI[s0]);
	  if(MI[s0] > themax) {
	    themax = MI[s0];
	    thethresh = s0;
	  }
	} // Loop over splits
	if(themax > bestmax) {
	  bestmax = themax;
	  bestind = d;
	  bestthresh = thethresh;
	}
      } // End loop over dimensions
      // Build the weak learner
      f[i] = bestind;
      t[i] = bestthresh;
      e[i] = build_decoder(ds,i);
      printf("Size of decoder dictionary %lu.\n",dec.size());
    } // End loop over classifiers
  } // End training function

  double build_decoder(dataset & ds, int i) {
    codeword c;
    dec.clear(); // Clear the dictionary
    
    // Loop over instances in dataset
    for(int s=0, xs=0; s<ds.num_instances; s++, xs+=ds.num_dimensions) {
      // Get codeword index for each data point
      c.reset();
      for(int j=0; j<i; j++) { if(ds.X[xs+f[j]] > t[j]) { c[j] = 1; } }
      unsigned char y = ds.y[s];

      auto it = dec.find(c);
      if(it == dec.end()) {
	rect r(ds.num_classes);
	r.num = 1;
	r.label = y;
	r.prob[y] = 1;
	dec.insert({c,r});
      } else {
	it->second.num++;
	it->second.label = y;
	it->second.prob[y]++;
      }
    } // End loop over instances in dataset
    return 0.0; // Return error
  } // End build_decoder function
};

#endif
