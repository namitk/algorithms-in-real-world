#ifndef SVMBASE_H
#define SVMBASE_H

#include <iostream>
#include <vector>
using namespace std;

struct sparse_vector_form {
  vector<int> id;
  vector<float> value;  
};

class svm_base {
 public:
  int num_examples;

  vector<int> labels;  
  float *hyperplane;
  float threshold;

  float two_sigma_squared;

  vector<sparse_vector_form> sparse_points;
  float *precomputed_self_dot_product;

  // reads a single example and returns it. At the same time, updates the number of features if a feature with id greater than current value of num_features is found
  sparse_vector_form read_example(istringstream& ss, int& num_features);

  // reads all examples. Returns the maximum feature index it encountered while reading all the examples. This should be the number of features in the data
  int read_data(ifstream& file);  

  // returns the point evaluated in the separting hyperplane equation (w.x-b)
  float classify_example_linear(int example_id);
  float classify_example_linear(sparse_vector_form svf);

  // returns the dot product of the two examples
  float dot_product(sparse_vector_form svf1, sparse_vector_form svf2);
  float dot_product(int ex1id, int ex2id);
  float dot_product(sparse_vector_form svf1, int point_index);

  // pre-compute norm of all points (self dot-products)
  void precompute_norm();

  // rbf kernel function
  float rbf_kernel(int i1, int i2);
  float rbf_kernel(sparse_vector_form svf, int point_index);
};

#endif
