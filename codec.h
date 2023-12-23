#ifndef CODEC_H
#define CODEC_H

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

class codeword {
 public:
  std::bitset<NUM_CLASSIFIERS> c;
  std::string s;
  
  codeword(int num_bits, int num_char) : s(num_char,'\0') { c.reset(); }
  ~codeword(){}

  void set(int i) { c[i] = 1; }
  void unset(int i) { c[i] = 0; }
  void zero(void) { c.reset(); }
  std::string & to_string(void) {
    for(int i=0, k=0; i<s.length(); i++) {
      char a = 0;
      for(int j=0; j<8 && k<c.size(); j++, k++) {
	char b = c[k];
	a |= b << j;
      }
      s[i] = a;
    }
    return s;
  }
};

// Class for codeword management
class codeword_builder {
 public:
  int num_bits;
  int num_char;
  codeword_builder(int num) {
    num_char = (num / 8) + ((num % 8) > 0);
  }
  ~codeword_builder() {}
  codeword make() { return codeword(num_bits, num_char); }
};

class codec {
 public:
  // Encoder parameters
  int num_classifiers;
  unsigned char* t; // thresholds 0, 1, 2, ..., 254
  int *f; // feature indexes
  float *e; // error rate as encoder is grown
  
  // Decoder parameters
  std::unordered_map<std::string, rect> dec;

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
    codeword_builder cb(num_classifiers); // Initalized to all zero bits
    codeword c = cb.make();
    std::string cs;

    // Create empty dictionaries
    std::vector<int> Mj(ds.num_classes); // Use an array
    std::unordered_map<std::string,int> Nc; // Use a dictionary
    std::vector<std::unordered_map<std::string,int>> Ncj; // Use an array of dictionaries
    for(int j=0; j<ds.num_classes; j++) {
      std::unordered_map<std::string,int> unmap;
      Ncj.push_back(unmap);
    }
    
    // Loop over classifiers
    for(int i=0; i<num_classifiers; i++) {
      printf("Working on classifier %3d.\n",i);
      double bestmax = -1e6;
      int bestind = 0;
      unsigned char bestthresh = 0;

      // Loop over dimensions
      for(int d=0; d<ds.num_dimensions; d++) {

	// Initialize all the array and dictionary data structures
	Nc.clear();
        for(int j=0; j<ds.num_classes; j++) {
	  Mj[j] = 0;
	  Ncj[j].clear();
	}

	// Initialize the dictionary counts
	for(int s=0, xs=0; s<ds.num_instances; s++, xs+=ds.num_dimensions) {
	  // Get codeword index for each data point
	  c.zero();
	  for(int j=0; j<i; j++) { if(ds.X[xs+f[j]] > t[j]) { c.set(j); } }

	  // Append a 1 to the codeword
	  c.set(i);
	  std::string cs = c.to_string();
	  unsigned char y = ds.y[s];
	  // Index the dictionaries and increment counts
	  Nc[cs]++;
	  Mj[y]++;
	  Ncj[y][cs]++;
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
	    c.zero();
	    for(int j=0; j<i; j++) { if(ds.X[ds.Xyi[s1].i*ds.num_dimensions + f[j]] > t[j]) { c.set(j); } }

	    // 2. Append a 1 to the codeword
	    c.set(i);
	    cs = c.to_string();
	    unsigned char y = ds.y[s1];

	    // 3. Index the dictionaries and decrement the counts
	    int Ncval = Nc[cs] - 1;
	    if(Ncval == 0) { Nc.erase(cs); } // Erase dictionary element
	    Mj[y]--; // Decrement array element
	    int Ncjval = Ncj[y][cs] - 1;
	    if(Ncjval == 0) { Ncj[y].erase(cs); } // Erase dictionary element

	    // 4. Append a 0 to the codeword
	    c.unset(i);
	    cs = c.to_string();
            
	    // 5. Index the dictionaries and increment the counts
            Nc[cs]++;
	    Mj[y]++;
	    Ncj[y][cs]++;

	    s1 += ds.num_dimensions;
	  } // Loop over same values
          // 6. Compute mutual information
	  for(int j=0; j<ds.num_classes; j++) {
	    int Mjval = Mj[j];
	    for(auto it = Ncj[j].begin(); it != Ncj[j].end(); it++) {
	      cs = it->first;
	      int Ncjval = it->second;
	      int Ncval = Nc[cs];
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
    codeword_builder cb(num_classifiers); // Initalized to all zero bits
    codeword c = cb.make();
    dec.clear(); // Clear the dictionary
    
    // Loop over instances in dataset
    for(int s=0, xs=0; s<ds.num_instances; s++, xs+=ds.num_dimensions) {
      // Get codeword index for each data point
      c.zero();
      for(int j=0; j<i; j++) { if(ds.X[xs+f[j]] > t[j]) { c.set(j); } }
      std::string cs = c.to_string();
      unsigned char y = ds.y[s];

      auto it = dec.find(cs);
      if(it == dec.end()) {
	rect r(ds.num_classes);
	r.num = 1;
	r.label = y;
	r.prob[y] = 1;
	dec[cs] = r;
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
