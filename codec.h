#ifndef CODEC_H
#define CODEC_H

#include <stdio.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include "dataset.h"

// Struct for tracking maximum metric (Mutual Information), feature index, and threshold
class stump {
 public:
  double me; // metric value or error
  int f; // feature index
  unsigned char t; // threshold
  stump() : me(-1e6), f(0), t(0) {}
  ~stump() {}
  void reset(void) { me = -1e6; f = 0; t = 0; }
};

typedef std::pair<unsigned char, unsigned char> lims;

// Class to manage rectangle construction
class rect {
 public:
  std::unordered_map<int, lims> dims; // Dimensions
};

// Class for elements of decoder dictionary
class rect_stat {
 public:
  std::vector<int> prob; // number of points from each class
  int num; // total number of points
  unsigned char label; // class label
  rect* r; // Rectangle

  rect_stat() : r(NULL) {}
  rect_stat(int num) : prob(num, 0), r(NULL) {}
  ~rect_stat() {}
  void reset(void) { std::fill(prob.begin(), prob.end(), 0); }
};

class codec {
 public:
  // Encoder parameters
  int num_classes;
  std::vector<stump> enc;
  // Thresholds 0, 1, 2, ..., 254
  // Feature indexes
  // Error rate as encoder is grown
  
  // Decoder parameters
  std::unordered_map<codeword, rect_stat, hash> dec;

  // Working variables/dictionaries for training (not saved or loaded)
  std::vector<int> Mj; // Use an array
  std::unordered_map<codeword,int,hash> Nc; // Use a dictionary
  std::vector<std::unordered_map<codeword,int,hash>> Ncj; // Use an array of dictionaries

  
  void save(char* filename, int num) {
    FILE* fid = fopen(filename,"wb");
    // Write out encoder parameters
    int num_classifiers = enc.size();
    fwrite(&num_classifiers,sizeof(num_classifiers),1,fid); // Number of classifiers in the container
    for(int i=0; i<num_classifiers; i++) {
      fwrite(&(enc[i].f),sizeof(enc[i].f),1,fid); // Feature indexes
      fwrite(&(enc[i].t),sizeof(enc[i].t),1,fid); // Thresholds
      fwrite(&(enc[i].me),sizeof(enc[i].me),1,fid); // Errors
    }
    // Write out decoder parameters
    int num_bits = dec.size();
    fwrite(&num_bits,sizeof(num_bits),1,fid); // Number of bits in the codeword
    num = dec.size(); // Size of dictionary
    fwrite(&num,sizeof(num),1,fid); // Size of dictionary
    for(auto & it : dec) {
      std::string cs = it.first.to_string(); // String version of codeword
      if(cs.size() != num_classifiers) {
	fprintf(stderr,"Houston, we have a problem with codeword strings. %lu %d\n",cs.length(),num_classifiers);
      }
      fwrite(cs.c_str(),sizeof(char),num_classifiers,fid);
      num = num_classes; // Number of classes = size of probability array
      fwrite(&num,sizeof(num),1,fid); // Size probability array
      fwrite(it.second.prob.data(),sizeof(int),num_classes,fid); // Probability array
      num = it.second.num; // Total points assigned to this rectangle
      fwrite(&num,sizeof(num),1,fid); // Total points assigned to this rectangle
      unsigned char val = it.second.label; // Label of this rectangle
      fwrite(&val,sizeof(val),1,fid); // Label of this rectangle
    }
    fclose(fid);
  } // End of save function

  void load(char* filename) {
    // This function assumes that the data structure is starting out "fresh",
    // and that no clearing of data structures is needed.  This function
    // goes directly and loads and fills an empty (assumption) data structure.
    FILE* fid = fopen(filename,"rb");
    // Read in encoder parameters
    int num_classifiers;
    fread(&num_classifiers,sizeof(num_classifiers),1,fid); // Number of classifiers in container
    enc.resize(num_classifiers);
    for(int i=0; i<num_classifiers; i++) {
      stump s;
      fread(&(s.f),sizeof(s.f),1,fid);
      fread(&(s.t),sizeof(s.t),1,fid);
      fread(&(s.me),sizeof(s.me),1,fid);
    }
    
    // Read in decoder parameters
    int num_bits, num;
    fread(&num_bits,sizeof(num_bits),1,fid); // Number of bits in the codeword
    char* cs = new char [num_bits]; // Allocate codeword string
    fread(&num,sizeof(num),1,fid); // Size of dictionary
    for(int i=0; i<num; i++) {
      fread(cs,sizeof(char),num_bits,fid);
      codeword c; // Initialized to all zero bits
      for(int j=0; j<num_bits; j++) {
	if(cs[j] == '1') {
	  c[j] = 1; // Set the appropriate bits to 1
	}
      }
      fread(&num_classes,sizeof(num_classes),1,fid); // Size of probability array
      rect_stat r(num_classes);
      fread(r.prob.data(),sizeof(int),num_classes,fid);
      fread(&(r.num),sizeof(r.num),1,fid);
      fread(&(r.label),sizeof(r.label),1,fid);

      // Add this rect to decoder dictionary
      dec.insert({c,r});
    }
    delete [] cs;
    fclose(fid);
  } // End of load function
  
  
  codec(int n_classes) {
    num_classes = n_classes;
    // Set up working variables/dictionaries
    Mj.resize(num_classes);
    Ncj.resize(num_classes);
    for(int j=0; j<num_classes; j++) {
      std::unordered_map<codeword,int,hash> unmap;
      Ncj.push_back(unmap);
    }
  }

  ~codec() {}

  void init_dictionaries(dataset &ds) {
    int i = enc.size();
    
    num_classes = ds.num_classes;
    // Initialize all the array and dictionary data structures
    Nc.clear(); // Clear the dictionary
    for(int j=0; j<ds.num_classes; j++) { // One dictionary for each class label
      Mj[j] = 0; // Initialize array elements to zero
      Ncj[j].clear(); // Clear the dictionaries
    }

    // Initialize the dictionary counts
    for(int s=0, xs=0; s<ds.num_instances; s++, xs+=ds.num_dimensions) {
      // Get the codeword index for this data point
      codeword & c = ds.C[s];
	  
      // Append a 1 to the codeword
      c[i] = 1;
      unsigned char y = ds.y[s];
      // Index the dictionaries and increment counts
      auto it = Nc.find(c);
      if(it == Nc.end()) {
	Nc.insert({c,1});
      } else {
	(it->second)++;
      }
      Mj[y]++;
      it = Ncj[y].find(c);
      if(it == Ncj[y].end()) {
	Ncj[y].insert({c,1});
      } else {
	(it->second)++;
      }
    } // End loop over instances (initialization of dictionary counts)
  }

  stump sweep_thresholds(dataset &ds, int d) {
    int i = enc.size();
    
    num_classes = ds.num_classes;
    // Sweep over all the split points
    stump themax;
    std::vector<double> MI(ds.num_splits,0.0); // Mutual information array
    int s1 = 0;
    int s1srt = ds.Xyi[s1].i;
    for(int s0=0; s0<ds.num_splits; s0++) {
      while(s1 < ds.num_instances && ds.X[s1srt*ds.num_dimensions + d] <= s0) {
	// 1. Get the codeword
	codeword & c = ds.C[s1srt];

	// 2. Append a 1 to the codeword
	c[i] = 1;
	unsigned char y = ds.y[s1srt];

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

#if 0
      // Analyze the dictionaries
      for(int k=0; k<ds.num_classes; k++) {
	printf("Mj[%2d] = %d\n",k,Mj[k]);
      }
      for(auto & k: Nc) {
	std::cout << k.first << " | " << k.second << '\n';
      }
      for(int k=0; k<ds.num_classes; k++) {
	for(auto & kk: Ncj[k]) {
	  std::cout << s0 << " " << k << " = " << kk.first << " | " << kk.second << '\n';
	}
      }
#endif
	  
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
	  
      if(MI[s0] > themax.me) {
	themax.me = MI[s0];
	themax.t = s0;
	themax.f = d;
      }
    } // Loop over splits

    return themax;
  }

  void train_next(dataset &ds) {
    int i = enc.size();
    
    if(i >= MAX_NUM_BITS) {
      fprintf(stderr,"ERROR: Need more bits!\n");
      return;
    }
    
    stump bestmax;

    // Loop over dimensions
    for(int d=0; d<ds.num_dimensions; d++) {

      // Initialize the dictionary data structures to store counts for this dimension
      init_dictionaries(ds);

      // Sort current dimension
      ds.sort_dimension(d);

      stump themax = sweep_thresholds(ds,d);

      if(themax.me > bestmax.me) { bestmax = themax; }
    } // End loop over dimensions
    // Build the weak learner

    // Encoder size grows by one
    enc.push_back(bestmax); // Save the best classifier by extending the encoder
    build_decoder(ds); // Build the decoder
    enc[i].me = compute_train_error(ds); // Save the error on the training set
    printf("Dictionary size = %lu\n",dec.size());
    printf("%3d: error rate = %f, x[%3d] <= %3d\n",i,enc[i].me,enc[i].f,enc[i].t);
  } // End train_next function
  
  void train_batch(dataset & ds, int num_classifiers) {
    // Loop over classifiers
    for(int i=0; i<num_classifiers; i++) {
      train_next(ds);
    } // End loop over classifiers
  } // End train_batch function

  void build_decoder(dataset & ds) {
    int i = enc.size() - 1;
    
    dec.clear(); // Clear the dictionary
    
    // Loop over instances in dataset
    for(int s=0, xs=0; s<ds.num_instances; s++, xs+=ds.num_dimensions) {
      // Compute codeword index for each data point
      codeword & c = ds.C[s];
      if(ds.X[xs+enc[i].f] > enc[i].t) {
	c[i] = 1;
      }
      unsigned char y = ds.y[s];

      auto it = dec.find(c);
      if(it == dec.end()) {
	rect_stat r(ds.num_classes);
	r.num = 1;
	r.prob[y] = 1;
	dec.insert({c,r});
      } else {
	it->second.num++;
	it->second.prob[y]++;
      }
    } // End loop over instances in dataset
  } // End build_decoder function

  double compute_train_error(dataset & ds) {
    int err = 0;
    // Loop over decoder dictionary entries
    for(auto & it : dec) {
      // Choose the class label by maximum count
      unsigned char y = 0;
      int themax = it.second.prob[y];

      // Print the dictionary element
      //printf("[%4d ",it.second.prob[0]);
      
      for(int j=1; j<ds.num_classes; j++) {
	//printf("%4d ",it.second.prob[j]);
      
	if(it.second.prob[j] > themax) {
	  themax = it.second.prob[j];
	  y = j;
	}
      }

      it.second.label = y;
      err += it.second.num - it.second.prob[y];
      //printf("], %5d, %d, %s\n",it.second.num,it.second.label,it.first.to_string().c_str());
    } // End loop over decoder dictionary entries
    return ((double)err) / ds.num_instances;
  } // End of compute_error function

  double compute_test_error(dataset & ds) {
    int err = 0;
    int miss = 0;
    // Loop over the dataset
    for(int s=0, xs=0; s<ds.num_instances; s++, xs+=ds.num_dimensions) {
      // Classify each feature vector and construct the codeword
      codeword c;
      int i = 0;
      for(auto & e : enc) { if(ds.X[xs+e.f] > e.t) { c[i] = 1; } i++; }

      // Look up the codeword in the decoder
      auto d = dec.find(c);
      if(d != dec.end()) {
	//printf("%d =?= %d\n",d->second.label,ds.y[s]);
	if(d->second.label != ds.y[s]) {
	  err++;
	}
      } else {
	miss++;
      }
    }
    double err_rate = ((double)err) / ds.num_instances;
    double miss_rate = ((double)miss) / ds.num_instances;
    printf("Test error rate = %f, \tmiss rate = %f\n",err_rate,miss_rate);
    return err_rate;
  } // End of compute_test_error function

  void build_rectangles(void) {
    for(auto & it : dec) { // Loop over the dictionary elements
      rect r; // Make a new rectangle for each dictionary elememnt
      for(int i=0; i<enc.size(); i++) { // Loop over bits--one for each encoder
	lims p0 = (0, enc[i].t);    // Interval for this threshold - low side
	lims p1 = (enc[i].t+1,255); // Interval for this threshold - high side

	auto l = r.dims.find(enc[i].f); // Is this feature index in the dims dictionary?
	if(l == r.dims.end()) { // Dimension does not exist
	  if(it.first[i]) { // 1 bit
	    lims q1 = (enc[i].t+1, 255); // Make a generic limits pair
	    r.dims.insert({enc[i].f,q1});
	  } else {
	    lims q0 = (0, enc[i].t); // Make a generic limits pair
	    r.dims.insert({enc[i].f,q0});
	  }
	} else { // Dimension does exist
	  // Compute the intersection
	  if(it.first[i]) { // 1 bit
	    // Lower limit is the maximum of two arguments
	    l->second.first = l->second.first > enc[i].t+1 ? l->second.first : enc[i].t+1;
	  } else {
	    /// Upper limit is the minimum of two arguments
	    l->second.second = l->second.second < enc[i].t ? l->second.second : enc[i].t;
	  }
	}
      }
    }
  }
  
};

#endif
