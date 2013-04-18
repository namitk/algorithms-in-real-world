#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <functional>
#include <unistd.h>
#include <sstream>
#include <vector>
#include <fstream>
#include <cassert>

#include "svm-predict.h"

using namespace std;

svm_predict::svm_predict(char* test_file,
			 char* model_file) {
  this->test_file = test_file;
  this->model_file = model_file;
  read_svm();     

  if (is_linear_kernel) {
    classify_example = &svm_predict::classify_example_linear;
    kernel_function = &svm_base::dot_product;
  } else {
    assert(num_examples == num_support_vectors);
    if (num_examples != num_support_vectors) {
      cout<<"Error reading data file..."<<endl;
      exit(-1);
    }
    classify_example = &svm_predict::classify_example_nonlinear;
    precompute_norm();    
    kernel_function = &svm_predict::rbf_kernel;      
  }
}

void svm_predict::read_svm() {
  ifstream file(model_file);
  file >> num_features;  
  cerr<<num_features<<endl;
  file >> is_linear_kernel;
  file >> threshold;
  if (is_linear_kernel) {
    hyperplane = new float[num_features];
    for (int i=0; i<num_features; i++)
      file >> hyperplane[i];
      } else {
  file >> two_sigma_squared;
    file >> num_support_vectors;
    sv_alphas = new float[num_support_vectors];
    for (int i=0; i<num_support_vectors; i++)
      file >> sv_alphas[i];   
    string skip_newline, line, token, tok;
    getline(file, skip_newline, '\n');
    read_data(file);
  }    
}
 
float svm_predict::classify_example_nonlinear(sparse_vector_form svf) {
  float s = 0.;
  for (int i=0; i<num_support_vectors; i++)    
    s += sv_alphas[i]*labels[i]*(this->*kernel_function)(svf, i);
  return s - threshold;
}

float svm_predict::rbf_kernel(sparse_vector_form svf, int point_index) {
  float s = dot_product(svf, point_index);
  s *= -2;
  s += dot_product(svf, svf) + precomputed_self_dot_product[point_index];
  return exp(-s/two_sigma_squared);
}

float svm_predict::test() {
  int num_test_examples;
  string line, token;
  ifstream file(test_file);
  int dummy = 0;
  int num_misclassified = 0;
  if (file.is_open()) {
    for (num_test_examples = 0; getline(file, line); num_test_examples++) {
      istringstream ss(line);
      getline(ss, token, ' ');
      int label = atoi(token.c_str());
      sparse_vector_form svf = read_example(ss, dummy);
      float result = (this->*classify_example)(svf);
      if ((result > 0) != (label == 1))     
	num_misclassified++;
    }
    return (num_test_examples - num_misclassified)*100.0/num_test_examples;
  }
}

int main(int argc, char** argv) {  
  char* test_file;
  char* model_file = "svm.model";

  // read parameters
  extern char* optarg;
  extern int optind;
  int c;
  int err_flag = 0;
  bool test_file_specified = false;
  while ((c=getopt(argc, argv, "f:m:")) != -1) {
    switch (c) {   
    case 'f':
      test_file_specified = true;
      test_file = optarg;
      break;
    case 'm':
      model_file = optarg;
      break;
    case '?':
      err_flag++;
    }    
  }
  if (!test_file_specified) {
    cout<<"Test file must be specified"<<endl;
    err_flag++;
  }    
  if (err_flag || optind < argc) {
    cerr<< "usage: "<<argv[0]<<" "<<endl<<
      "-f test_file"<<endl<<
      "-m model_file"<<endl;
      exit(2);
  }

  svm_predict svm(test_file,
		  model_file);
  cout<<svm.test()<<endl;
  return 0;
}
