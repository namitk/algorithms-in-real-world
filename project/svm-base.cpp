#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <cmath>

#include "svm-base.h"

using namespace std;

// reads a single example. At the same time, updates the number of features if a feature with id greater than current value of num_features is found
sparse_vector_form svm_base::read_example(istringstream& ss, int& num_features) {
  sparse_vector_form svf;    
  string token, tok;
  while(getline(ss, token, ' ')) {
    istringstream tk(token);     
    getline(tk, tok, ':');
    int index = atoi(tok.c_str());
    if (index > num_features)
      num_features = index + 1;
    svf.id.push_back(index);	  
    getline(tk, tok, ':');
    svf.value.push_back(atof(tok.c_str()));
  }
  return svf;
}

// reads all examples. Returns the maximum feature index it encountered while reading all the examples. This should be the number of features in the data
int svm_base::read_data(ifstream& file) {    
  int num_features = 0;
  string line, token;        
  if (file.is_open()) {
    for (this->num_examples = 0; getline(file, line); this->num_examples++) {
      istringstream ss(line);	
      getline(ss, token, ' ');
      labels.push_back(atoi(token.c_str()));
      sparse_points.push_back(read_example(ss, num_features));
    }
  }
  return num_features;
}

// returns the point evaluated in the separting hyperplane equation (w.x-b)
float svm_base::classify_example_linear(int example_id) {
  float s = 0.;
  for (unsigned int i=0; i<sparse_points[example_id].value.size(); i++)
    s += hyperplane[sparse_points[example_id].id[i]]*sparse_points[example_id].value[i];
  return s - threshold;
}

float svm_base::classify_example_linear(sparse_vector_form svf) {
  float s = 0.;
  for (unsigned int i=0; i<svf.value.size(); i++)
    s += hyperplane[svf.id[i]]*svf.value[i];
  return s - threshold;
}

// returns the dot product of the two examples 
float svm_base::dot_product(sparse_vector_form svf1, sparse_vector_form svf2) {
  int p1 = 0, p2 = 0;
  float dot = 0.;
  int num1 = svf1.value.size();
  int num2 = svf2.value.size();
  while (p1 < num1 && p2 < num2) {
    int id1 = svf1.id[p1];
    int id2 = svf2.id[p2];
    if (id1==id2) {
      dot += svf1.value[p1]*svf2.value[p2];
      p1++;
      p2++;
    } else if (id1 > id2)
      p2++;
    else
      p1++;
  }
  return dot;
}

float svm_base::dot_product(int ex1id, int ex2id) {
  sparse_vector_form svf1 = sparse_points[ex1id];
  sparse_vector_form svf2 = sparse_points[ex2id];
  return dot_product(svf1, svf2);
}

float svm_base::dot_product(sparse_vector_form svf1, int point_index) {
  sparse_vector_form svf2 = sparse_points[point_index];
  return dot_product(svf1, svf2);
}

void svm_base::precompute_norm() {
  precomputed_self_dot_product = new float[num_examples];
  for (int i=0; i<num_examples; i++)
    precomputed_self_dot_product[i] = dot_product(i, i);
}

float svm_base::rbf_kernel(int i1, int i2) {
  float s = dot_product(i1, i2);
  s *= -2;
  s += precomputed_self_dot_product[i1] + precomputed_self_dot_product[i2];
  return exp(-s/two_sigma_squared);
}

float svm_base::rbf_kernel(sparse_vector_form svf, int point_index) {
  float s = dot_product(svf, point_index);
  s *= -2;
  s += dot_product(svf, svf) + precomputed_self_dot_product[point_index];
  return exp(-s/two_sigma_squared);
}
