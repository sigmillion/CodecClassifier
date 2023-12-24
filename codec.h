#ifndef CODEC_H
#define CODEC_H

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include "dataset.h"

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
    //codeword c;
    unsigned char y;
    
    // Create empty dictionaries
    std::vector<int> Mj(ds.num_classes); // Use an array
    std::unordered_map<codeword,int,hash> Nc; // Use a dictionary
    std::vector<std::unordered_map<codeword,int,hash>> Ncj; // Use an array of dictionaries
    for(int j=0; j<ds.num_classes; j++) {
      std::unordered_map<codeword,int,hash> unmap;
      Ncj.push_back(unmap);
    }

    // Loop over classifiers
    for(int i=0; i<num_classifiers; i++) {
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
	  // Compute codeword index for each data point
	  //c.reset();
	  //for(int j=0; j<i; j++) { if(ds.X[xs+f[j]] > t[j]) { c[j] = 1; } }

	  codeword & c = ds.C[s];
	  
	  // Append a 1 to the codeword
	  c[i] = 1;
	  y = ds.y[s];
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
	int s1srt = ds.Xyi[s1].i;
	for(int s0=0; s0<ds.num_splits; s0++) {
	  while(s1 < ds.num_instances && ds.X[s1srt*ds.num_dimensions + d] <= s0) {
	    // 1. Compute codeword for x(s)
	    //c.reset();
	    //for(int j=0; j<i; j++) { if(ds.X[ds.Xyi[s1].i*ds.num_dimensions + f[j]] > t[j]) { c[j] =1; } }

	    codeword & c = ds.C[s1srt];

	    // 2. Append a 1 to the codeword
	    c[i] = 1;
	    y = ds.y[s1srt];

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

	    s1++;
	    s1srt = ds.Xyi[s1].i;
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
	      codeword c = it->first;
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
      build_decoder(ds,i);
      e[i] = compute_error(ds,i);
      printf("Dictionary size = %lu\n",dec.size());
      printf("%3d: error rate = %f, x[%3d] <= %3d\n",i,e[i],f[i],t[i]);
    } // End loop over classifiers
  } // End training function

  void build_decoder(dataset & ds, int i) {
    unsigned char y;
    dec.clear(); // Clear the dictionary
    
    // Loop over instances in dataset
    for(int s=0, xs=0; s<ds.num_instances; s++, xs+=ds.num_dimensions) {
      // Compute codeword index for each data point
      //c.reset();
      //for(int j=0; j<=i; j++) { if(ds.X[xs+f[j]] > t[j]) { c[j] = 1; } }
      // Update codeword.  The old way of computing the codeword was
      // computationally very wasteful.  Now we just look it up and set a bit.
      codeword & c = ds.C[s];
      if(ds.X[xs+f[i]] > t[i]) {
	//ds.C[s][i] = 1;
	c[i] = 1;
      }
      y = ds.y[s];

      //auto it = dec.find(ds.C[s]);
      auto it = dec.find(c);
      if(it == dec.end()) {
	rect r(ds.num_classes);
	r.num = 1;
	r.prob[y] = 1;
	//dec.insert({ds.C[s],r});
	dec.insert({c,r});
      } else {
	it->second.num++;
	it->second.prob[y]++;
      }
    } // End loop over instances in dataset
  } // End build_decoder function

  double compute_error(dataset & ds, int i) {
    int err = 0;
    // Loop over decoder dictionary entries
    for(auto it : dec) {
      // Choose the class label by maximum count
      unsigned char y = 0;
      int themax = it.second.prob[y];
      for(int j=1; j<ds.num_classes; j++) {
	if(it.second.prob[j] > themax) {
	  themax = it.second.prob[j];
	  y = j;
	}
      }
      it.second.label = y;
      err += it.second.num - it.second.prob[y];
    } // End loop over decoder dictionary entries
    return ((double)err) / ds.num_instances;
  } // End of compute_error function
};

#endif
