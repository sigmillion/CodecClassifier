#ifndef CODEC_H
#define CODEC_H

#include <stdio.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <random>
#include "dataset.h"

class myrand {
 public:
  std::mt19937 mt;
  myrand() : mt(time(NULL)) {}
  myrand(int seed) : mt(seed) {}
  ~myrand() {}
  int rand(int themax) {
    // Generates random numbers between 0, 1, 2, ..., themax-1
    return (mt() % themax);
  }
};

class graph_search_metric {
 public:
  int dist;
  unsigned char label;
  graph_search_metric(int d, unsigned char y) : dist(d), label(y) {}
  graph_search_metric() : dist(1000000) {}
  ~graph_search_metric() {}
};

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

class lims {
 public:
  unsigned char lower; // lower limit
  unsigned char upper; // upper limit
  int lower_bit; // index of codeword bit
  int upper_bit; // index of codeword bit
  lims() {}
  lims(int l, int u, int bl, int bu) : lower(l), upper(u), lower_bit(bl), upper_bit(bu) {}
  ~lims() {}
};

// Class to manage rectangle construction
class rect {
 public:
  std::unordered_map<int, lims> dims; // Dimensions
  void print(void) {
    for(auto & it : dims) {
      printf("Dim %3d: (%3d,%3d) bit low = %d bit high = %d\n",it.first,
	     it.second.lower,it.second.upper,it.second.lower_bit,it.second.upper_bit);
    }
    printf("================================================\n");
  }
};

// Class for elements of decoder dictionary
class rect_stat {
 public:
  std::vector<int> prob; // number of points from each class
  int num; // total number of points
  unsigned char label; // class label
  rect* r; // Rectangle

  rect_stat() : prob(10, 0), r(NULL) {}
  rect_stat(int num) : prob(num, 0), r(NULL) {}
  ~rect_stat() { if(r != NULL) { delete r; } }

  void reset(void) { std::fill(prob.begin(), prob.end(), 0); }
  void print(void) {
    if(r != NULL) {
      printf("Label=%d, [",label);
      for(auto & p : prob) { printf("%4d ",p); }
      printf("], %6d\n",num);
      r->print();
    }
  }
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

  void print(void) {
    printf("Encoder ==================================\n");
    int i = 0;
    for(auto & e : enc) {
      printf("%4d: x[%4d] <= %3d\n",i,e.f,e.t);
      i++;
    }
    printf("Decoder ==================================\n");
    for(auto & d : dec) {
      printf("cw = %s\n",d.first.to_string().c_str());
      d.second.print();
    }
  }
  
  void save(const char* filename) {
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
    int num_bits = enc.size();
    fwrite(&num_bits,sizeof(num_bits),1,fid); // Number of bits in the codeword
    int num = dec.size(); // Size of dictionary
    fwrite(&num,sizeof(num),1,fid); // Size of dictionary
    for(auto & it : dec) {
      for(int i=0; i<num_bits; i++) {
	if(it.first[i]) {
	  char a = '1';
	  fwrite(&a,sizeof(a),1,fid);
	} else {
	  char a = '0';
	  fwrite(&a,sizeof(a),1,fid);
	}
      }
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

  void load(const char* filename) {
    // This function assumes that the data structure is starting out "fresh",
    // and that no clearing of data structures is needed.  This function
    // goes directly and loads and fills an empty (assumption) data structure.
    FILE* fid = fopen(filename,"rb");
    // Read in encoder parameters
    int num_classifiers;
    fread(&num_classifiers,sizeof(num_classifiers),1,fid); // Number of classifiers in container
    //enc.resize(num_classifiers);
    for(int i=0; i<num_classifiers; i++) {
      stump s;
      fread(&(s.f),sizeof(s.f),1,fid);
      fread(&(s.t),sizeof(s.t),1,fid);
      fread(&(s.me),sizeof(s.me),1,fid);
      enc.push_back(s);
    }
    
    // Read in decoder parameters
    int num_bits, num;
    fread(&num_bits,sizeof(num_bits),1,fid); // Number of bits in the codeword
    fread(&num,sizeof(num),1,fid); // Size of dictionary
    for(int i=0; i<num; i++) {
      codeword c; // Initialized to all zero bits
      char a;
      for(int j=0; j<num_bits; j++) {
	fread(&a,sizeof(a),1,fid);
	if(a == '1') { c.set(j); }
      }
      fread(&num_classes,sizeof(num_classes),1,fid); // Size of probability array
      rect_stat r(num_classes);
      fread(r.prob.data(),sizeof(int),num_classes,fid);
      fread(&(r.num),sizeof(r.num),1,fid);
      fread(&(r.label),sizeof(r.label),1,fid);

      // Add this rect to decoder dictionary
      dec.insert({c,r});
    }
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

  void set_codewords(dataset & ds) {
    // Loop over the dataset
    for(int s=0, xs=0; s<ds.num_instances; s++, xs+=ds.num_dimensions) {
      // Classify each feature vector and construct the codeword
      //codeword c;
      codeword & c = ds.C[s];
      int i = 0;
      for(auto & e : enc) { if(ds.X[xs+e.f] > e.t) { c[i] = 1; } i++; }
    }    
  }
  
  double compute_test_error(dataset & ds) {
    int err = 0;
    int miss = 0;
    // Loop over the dataset
    for(int s=0, xs=0; s<ds.num_instances; s++, xs+=ds.num_dimensions) {
      // Classify each feature vector and construct the codeword
      //codeword c;
      codeword & c = ds.C[s];
      int i = 0;
      for(auto & e : enc) { if(ds.X[xs+e.f] > e.t) { c[i] = 1; } i++; }

      // Look up the codeword in the decoder
      auto d = dec.find(c);
      if(d != dec.end()) {
#if 0
	// Sanity test:  Does the given x live in this rectangle?  Yes, this checks out!
	for(auto dim : d->second.r->dims) {
	  fprintf(stderr,"[%3d <= %3d <= %3d] ",dim.second.lower,ds.X[xs+dim.first],dim.second.upper);
	  if(dim.second.lower <= ds.X[xs+dim.first] && ds.X[xs+dim.first] <= dim.second.upper) {
	    fprintf(stderr,"\n");
	  } else {
	    fprintf(stderr,"*\n");
	  }
	}
#endif
	if(d->second.label != ds.y[s]) {
	  err++;
	}
      } else {
	miss++;
#if 0
    // Write out the data
    FILE* fid = fopen("x_data.dat","ab");
    unsigned char* xd = ds.X+xs;
    int xlen = ds.num_dimensions;
    fwrite(xd,sizeof(unsigned char),xlen,fid);
    fclose(fid);
#endif
      }
    }
    double err_rate = ((double)err) / ds.num_instances;
    double miss_rate = ((double)miss) / ds.num_instances;
    printf("Test error rate = %f, \tmiss rate = %f\n",err_rate,miss_rate);
    return err_rate;
  } // End of compute_test_error function

  void build_rectangles(void) {
    for(auto & it : dec) { // Loop over the dictionary elements
      it.second.r = build_rect_from_codeword(it.first); // Make this rect_stat point to this rect
#if 0
      rect *r = new rect; // Make a new rectangle for each dictionary elememnt
      for(int i=0; i<enc.size(); i++) { // Loop over bits--one for each encoder
	auto l = r->dims.find(enc[i].f); // Is this feature index in the dims dictionary?
	if(l == r->dims.end()) { // Dimension does not exist
	  if(it.first[i]) { // 1 bit
	    lims q1 = lims(enc[i].t+1, 255, i, -1); // Make a generic limits pair
	    r->dims.insert({enc[i].f,q1});
	  } else {
	    lims q0 = lims(0, enc[i].t, -1, i); // Make a generic limits pair
	    r->dims.insert({enc[i].f,q0});
	  }
	} else { // Dimension does exist
	  // Compute the intersection
	  if(it.first[i]) { // 1 bit
	    // Lower limit is the maximum of two arguments
	    if(l->second.lower < enc[i].t+1) {
	      l->second.lower = enc[i].t+1;
	      l->second.lower_bit = i; // Save the bit index too
	    }
	  } else {
	    /// Upper limit is the minimum of two arguments
	    if(l->second.upper > enc[i].t) {
	      l->second.upper = enc[i].t;
	      l->second.upper_bit = i; // Save the bit index too
	    }
	  }
	} // End if dimension exists
      } // End loop over bits
      it.second.r = r; // Make this rect_stat point to this rect
#endif
    } // End loop over decoder dictionary elements
  } // End build_rectangles function

  rect* build_rect_from_codeword(const codeword & c) {
    rect* r = new rect; // Make a new rectangle for each dictionary elememnt
    for(int i=0; i<enc.size(); i++) { // Loop over bits--one for each encoder
      auto l = r->dims.find(enc[i].f); // Is this feature index in the dims dictionary?
      if(l == r->dims.end()) { // Dimension does not exist
	if(c[i]) { // 1 bit
	  lims q1 = lims(enc[i].t+1, 255, i, -1); // Make a generic limits pair
	  r->dims.insert({enc[i].f,q1});
	} else {
	  lims q0 = lims(0, enc[i].t, -1, i); // Make a generic limits pair
	  r->dims.insert({enc[i].f,q0});
	}
      } else { // Dimension does exist
	// Compute the intersection
	if(c[i]) { // 1 bit
	  // Lower limit is the maximum of two arguments
	  if(l->second.lower < enc[i].t+1) {
	    l->second.lower = enc[i].t+1;
	    l->second.lower_bit = i; // Save the bit index too
	  }
	} else {
	  /// Upper limit is the minimum of two arguments
	  if(l->second.upper > enc[i].t) {
	    l->second.upper = enc[i].t;
	    l->second.upper_bit = i; // Save the bit index too
	  }
	}
      } // End if dimension exists
    } // End loop over bits
    return r;
  } // End build_rectangles function
  
  // This function does (1) an encoding and (2) a decoding.  If the
  // codeword is not found in the decoder dictionary, then a depth
  // first search is initiated.  Searching through the region
  // adjacency graph continues until a valid rectangle is encountered.
  // The empty rectangles are labeled and added to the decoder dictionary
  // during the graph traversal.  This will speed up future predictions.
  void predict_and_fix(dataset & ds) {
    // Loop over the dataset
    for(int s=0, xs=0; s<ds.num_instances; s++, xs+=ds.num_dimensions) {
      // Classify each feature vector and construct the codeword
      codeword c = ds.C[s]; // Make a copy of the codeword
      int i = 0; c.reset();
      for(auto & e : enc) { if(ds.X[xs+e.f] > e.t) { c[i] = 1; } i++; }
      std::vector<unsigned char> x(ds.num_dimensions); // Make a vector copy of the feature data
      for(int i=0; i<ds.num_dimensions; i++) { x[i] = ds.X[xs+i]; }

      graph_search_metric gsm = recursive_graph_search(x,c,0);
      // ds.y[s] = gsm.label; // Don't do this or else you get zero error rate.
    }
    printf("Dictionary size == %lu\n",dec.size());
  } // End of compute_test_error function

  graph_search_metric recursive_graph_search(std::vector<unsigned char> & x, codeword & c, int dist) {
    //===============================================
    // Look up the codeword in the decoder and return label if found
    auto d = dec.find(c);
    if(d != dec.end()) { // Found it in the dictionary
      return graph_search_metric(dist,d->second.label); // Return the label
    }
    //===============================================
    // Did not find the codeword
    //fprintf(stderr,"Miss ... \n");

    // Codeword check
    //codeword ctmp;
    //int i = 0;
    //for(auto & e : enc) { if(x[e.f] > e.t) { ctmp[i] = 1; } i++; }
    //fprintf(stderr,">>%s\n",ctmp.to_string().c_str());
    //fprintf(stderr,">>%s\n",c.to_string().c_str());
    
    // 1. Build rect_stat and rect structures
    rect_stat rs;
    auto it = dec.insert({c,rs});
    it.first->second.r = build_rect_from_codeword(c); // Put this pointer into the rect_stat in the dictionary
                                                // so that it doesn't get deleted when rs is deconstructed
                                                // when this function terminates.

    //for(auto & d : it.first->second.r->dims) {
    //  fprintf(stderr,"dim %3d (%3d) [%3d <= %3d <= %3d] (%3d)\n",
    //  d.first, d.second.lower_bit, d.second.lower, x[d.first], d.second.upper,
    //  d.second.upper_bit);
    //}
    
    // 2. Begin depth-first search of region adjacency graph
    // using rectangles and codewords
    graph_search_metric bestgsm, gsm;
    for(auto & l : it.first->second.r->dims) { // Loop over the limits
      // l is a dimension for the rectangle currently in
      if(l.second.lower_bit != -1) { // Valid lower limit
	unsigned char xold = x[l.first];
	x[l.first] = l.second.lower - 1; // -1 to step into next rectangle below
	c[l.second.lower_bit] = !c[l.second.lower_bit]; // Flip the bit
	int dist_increment = xold - x[l.first];
	//*** The codeword and x don't match at this point!!!!
	//ctmp.reset(); i = 0;
	//for(auto & e : enc) { if(x[e.f] > e.t) { ctmp[i] = 1; } i++; }
	//fprintf(stderr,"%2d %3d||%s\n",l.second.lower_bit,i,ctmp.to_string().c_str());
	//fprintf(stderr,"      ||%s\n",c.to_string().c_str());
        //fprintf(stderr,"lower %d + %d\n",dist,dist_increment);
        //***
	gsm = recursive_graph_search(x,c, dist+dist_increment);
	c[l.second.lower_bit] = !c[l.second.lower_bit]; // Flip the bit back to the way it was
	x[l.first] = xold; // Put x back the way it was
	if(gsm.dist < bestgsm.dist) {
	  bestgsm = gsm;
	}
      }
      if(l.second.upper_bit != -1) { // Valid upper limit
	unsigned char xold = x[l.first];
	x[l.first] = l.second.upper + 1; // +1 to step into the next rectangle above
	c[l.second.upper_bit] = !c[l.second.upper_bit]; // Flip the bit
	int dist_increment = x[l.first] - xold;
	//***
	//ctmp.reset(); i = 0;
	//for(auto & e : enc) { if(x[e.f] > e.t) { ctmp[i] = 1; } i++; }
	//fprintf(stderr,"%2d %3d~~%s\n",l.second.upper_bit,i,ctmp.to_string().c_str());
	//fprintf(stderr,"      ~~%s\n",c.to_string().c_str());
	//fprintf(stderr,"upper %d + %d\n",dist,dist_increment);
        //***
	gsm = recursive_graph_search(x,c, dist+dist_increment);
	c[l.second.upper_bit] = !c[l.second.upper_bit]; // Flip the bit back
	x[l.first] = xold; // Pub x back the way it was
	if(gsm.dist < bestgsm.dist) {
	  bestgsm = gsm;
	}
      }
    }
    // 3. Set the label
    //fprintf(stderr,"distance %d\n",gsm.dist);
    //rs.label = bestgsm.label; // Set the label in this rect_stat
    // 4. Add rect_stat to the decoder dictionary
    //dec.insert({c,rs});
    it.first->second.label = bestgsm.label;

    return bestgsm;
  } // End recursive_graph_search function

  void testcode(void) {
    int num_data = 16;
    int num_dimensions = 784;
    FILE* fid = fopen("x_data.dat","rb");
    unsigned char* x = new unsigned char [num_data * num_dimensions];
    fread(x,sizeof(unsigned char),num_data * num_dimensions,fid);
    fclose(fid);

    unsigned char* xp=x;
    for(int j=0; j<1 /*num_data*/; j++, xp+=num_dimensions) {
      // Compute the codeword
      codeword c;
      int i = 0;
      for(auto & e : enc) { c.set(i,x[e.f] > e.t); i++; }
      fprintf(stderr,">>%s\n",c.to_string().c_str());
      std::vector<unsigned char> xv(num_dimensions);
      for(i=0; i<num_dimensions; i++) { xv[i] = x[i]; }
      graph_search_metric gsm = recursive_graph_search(xv,c,0);
    }
#if 0      
      // Get the dictionary element for this codeword
      auto it = dec.find(c);
      if(it != dec.end()) {
      } else {
	// Print the rectangle
	for(auto & d : it->second.r->dims) {
	  fprintf(stderr,"dim %3d (%3d) [%3d <= %3d <= %3d] (%3d)\n",
		  d.first, d.second.lower_bit, d.second.lower, xp[d.first], d.second.upper,
		  d.second.upper_bit);
	}
      }
#endif
    
    delete [] x;
  }

}; // End codec class definition

#endif
