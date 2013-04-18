#ifndef SVMPREDICT_H
#define SVMPREDICT_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>

#include "svm-base.h"

using namespace std;

class svm_predict : public svm_base {
 public:
  char* test_file;
  char* model_file;

  float two_sigma_squared;

  bool is_linear_kernel;
  int num_features;
  int num_support_vectors;

  float* sv_alphas;
  
  float (svm_predict::*classify_example)(sparse_vector_form);
  float (svm_predict::*kernel_function)(sparse_vector_form, int);

  svm_predict(char* test_file,
	      char* model_file);
  
  void read_svm();
  
  float classify_example_nonlinear(sparse_vector_form svf);

  float rbf_kernel(sparse_vector_form svf, int point_index);

  float test();
};

#endif
