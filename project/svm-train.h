#ifndef SVMTRAIN_H
#define SVMTRAIN_H

#include <iostream>
#include <string>
#include <vector>

#include "svm-base.h"

using namespace std;

class svm_train : public svm_base {
 public:
  char* data_file;
  char* model_file;

  float slack_penalty; // C
  float tolerance; 
  float epsilon;  
  float two_sigma_squared;

  float* alphas; // lagrange multipliers
  float delta_threshold;
  float* error_cache;
  
  float* precomputed_self_dot_product;   
  bool is_linear_kernel;

  int num_features; //number of features

  float (svm_train::*classify_example)(int);
  float (svm_train::*kernel_function)(int, int);

  svm_train(char* data_file,
	    char* model_file,		
	    float slack_penalty,
	    float tolerance,
	    float epsilon,
	    float two_sigma_squared,
	    bool is_linear_kernel); 

  float classify_example_nonlinear(int i);

  float rbf_kernel(int i1, int i2);

  int examine_example(int i1);

  int take_step(int i1, int i2);

  void train();

  void write_svm();
};

#endif
