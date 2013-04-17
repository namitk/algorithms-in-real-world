#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <unistd.h>
#include <cmath>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>

#include "svm-train.h"

using namespace std;

svm_train::svm_train(char* data_file,
		     char* model_file,		
		     float slack_penalty,
		     float tolerance,
		     float epsilon,
		     float two_sigma_squared,
		     bool is_linear_kernel) {
  this->data_file = data_file;
  this->model_file = model_file;

  this->slack_penalty = slack_penalty; // C
  this->tolerance = tolerance; 
  this->epsilon = epsilon;  
  this->two_sigma_squared = two_sigma_squared;
  this->is_linear_kernel = is_linear_kernel;
         
  ifstream file(data_file);
  this->num_features = read_data(file);
  if (num_features==0) {
    cout<<"Error reading data file..."<<endl;
    exit(-1);
  }

  alphas = new float[num_examples];
  fill(alphas, alphas+num_examples, 0.0);
   
  threshold = 0.0;
  error_cache = new float[num_examples];
  if (is_linear_kernel) {
    hyperplane = new float[num_features];
    fill(hyperplane, hyperplane+num_features, 0.0);
    classify_example = &svm_base::classify_example_linear;
    kernel_function = &svm_base::dot_product;
  } else {
    classify_example = &svm_train::classify_example_nonlinear;
    precomputed_self_dot_product = new float[num_examples];
    for (int i=0; i<num_examples; i++)
      precomputed_self_dot_product[i] = dot_product(i, i);
    kernel_function = &svm_train::rbf_kernel;      
  } 
}

float svm_train::classify_example_nonlinear(int k) {
  float s = 0.;
  for (int i=0; i<num_examples; i++)
    if (alphas[i] > 0)
      s += alphas[i]*labels[i]*(this->*kernel_function)(i, k);
  return s - threshold;
}

float svm_train::rbf_kernel(int i1, int i2) {
  float s = dot_product(i1, i2);
  s *= -2;
  s += precomputed_self_dot_product[i1] + precomputed_self_dot_product[i2];
  return exp(-s/two_sigma_squared);
}

int svm_train::examine_example(int i1) {
  float alpha1, E1, r1;
  int y1;
  y1 = labels[i1];
  alpha1 = alphas[i1];
    
  if (alpha1 > 0 && alpha1 < slack_penalty)
    E1 = error_cache[i1];
  else
    E1 = (this->*classify_example)(i1) - y1;

  r1 = y1*E1;
  if ((r1 < -tolerance && alpha1 < slack_penalty) || (r1 > tolerance && alpha1 > 0)) {
    // try argmax E1-E2
    {
      int k, i2;
      float tmax;
	
      for (i2=-1, tmax=0, k=0; k<num_examples; k++) {
	if (alphas[k] > 0 && alphas[k] < slack_penalty) {
	  float E2 = error_cache[k];
	  float temp = fabs(E1-E2);
	  if (temp > tmax) {
	    tmax = temp;
	    i2 = k;
	  }
	}
      }
      if (i2 >= 0)
	if (take_step(i1, i2))
	  return 1;
    }
    // try iterating through the non-bound examples
    {
      int k, k0, i2;
      for (k0 = (int) (drand48() * num_examples), k=k0; k<num_examples+k0; k++) {
	i2 = k % num_examples;
	if (alphas[i2] > 0 && alphas[i2] < slack_penalty) {
	  if (take_step(i1, i2))
	    return 1;
	}
      }
    }
    // try iterating through the entire training set
    {
      int k0, k, i2;
      for (k0 = (int) (drand48() * num_examples), k=k0; k<num_examples+k0; k++) {
	i2 = k%num_examples;
	if (take_step(i1, i2))
	  return 1;
      }
    }
  }
  return 0;
}

int svm_train::take_step(int i1, int i2) {
  int y1, y2, s;
  float alpha1_old, alpha2_old;
  float alpha1_new, alpha2_new;
  float E1, E2, L, H, k11, k22, k12, eta, Lobj, Hobj;

  if (i1==i2)
    return 0;
    
  // look up alpha1_old, y1, E1, alpha2_old, y2, E2
  alpha1_old = alphas[i1];
  y1 = labels[i1];
  if (alpha1_old > 0 && alpha1_old < slack_penalty)
    E1 = error_cache[i1];
  else
    E1 = (this->*classify_example)(i1) - y1;

  alpha2_old = alphas[i2];
  y2 = labels[i2];
  if (alpha2_old > 0 && alpha2_old < slack_penalty)
    E2 = error_cache[i2];
  else
    E2 = (this->*classify_example)(i2) - y2;    

  s = y1 * y2;
  // compute L, H
  if (y1 == y2) {
    float gamma = alpha1_old + alpha2_old;
    L = max((float)0., gamma - slack_penalty);
    H = min(slack_penalty, gamma);
  } else {
    float gamma = alpha2_old - alpha1_old;
    L = max((float)0., gamma);
    H = min(slack_penalty, slack_penalty + gamma);
  }
  if (L==H)
    return 0;

  // compute eta
  k11 = (this->*kernel_function)(i1, i1);
  k12 = (this->*kernel_function)(i1, i2);
  k22 = (this->*kernel_function)(i2, i2);
  eta = 2*k12 - k11 - k22;

  if (eta < 0) {
    alpha2_new = alpha2_old + y2*(E2-E1)/eta;
    if (alpha2_new < L)
      alpha2_new = L;
    else if (alpha2_new > H)
      alpha2_new = H;
  } else {
    // compute Lobj = objective function at alpha2=L. Similarly Hobj
    /*{
      float c1 = eta/2, c2 = y2*(E1-E2)-eta*alpha2_old;
      Lobj = c1*L*L + c2*L;
      Hobj = c1*H*H + c2*H;
      }*/
    float f1 = y1*(E1 + threshold) - alpha1_old*k11 - s*alpha2_old*k12;
    float f2 = y2*(E2 + threshold) - s*alpha1_old*k12 - alpha2_old*k22;
    float L1 = alpha1_old + s*(alpha2_old - L);
    float H1 = alpha1_old + s*(alpha2_old - H);
    float Lobj = L1*f1 + L*f2 + .5*L1*L1*k11 + .5*L*L*k22 + s*L*L1*k12;
    float Hobj = H1*f1 + H*f2 + .5*H1*H1*k11 + .5*H*H*k22 + s*H*H1*k12;
    if (Lobj < Hobj - epsilon)
      alpha2_new = L;
    else if (Lobj < Hobj + epsilon)
      alpha2_new = H;
    else
      alpha2_new = alpha2_old;
  }

  if (fabs(alpha2_new - alpha2_old) < epsilon*(alpha2_new + alpha2_old + epsilon))
    return 0;
    
  alpha1_new = alpha1_old - s*(alpha2_new - alpha2_old);
  if (alpha1_new < 0) {
    alpha2_new += s*alpha1_new;    
    alpha1_new = 0;
  } else if (alpha1_new > slack_penalty) {
    alpha2_new += s*(alpha1_new-slack_penalty);    
    alpha1_new = slack_penalty;
  }

  // update threshold to reflect change in lagrange multipliers 
  float b1 = threshold + E1 + y1*(alpha1_new-alpha1_old)*k11 + y2*(alpha2_new-alpha2_old)*k12;
  float b2 = threshold + E2 + y1*(alpha1_new-alpha1_old)*k12 + y2*(alpha2_new-alpha2_old)*k22;
  float threshold_new;
  if (alpha1_new > 0 && alpha1_new < slack_penalty)
    threshold_new = b1;
  else if (alpha2_new > 0 && alpha2_new < slack_penalty)
      threshold_new = b2;
  else 
      threshold_new = (b1+b2)/2;
  /* what is this */  
  delta_threshold = threshold_new - threshold;
  threshold = threshold_new;  

  float t1 = y1*(alpha1_new - alpha1_old);
  float t2 = y2*(alpha2_new-alpha2_old);

  // update weight vector to reflect change in alpha1 and alpha2, if linear SVM
  if (is_linear_kernel) {            
    int num1 = sparse_points[i1].value.size();
    for (int i=0; i<num1; i++)
      hyperplane[sparse_points[i1].id[i]] += t1*sparse_points[i1].value[i];
    int num2 = sparse_points[i2].value.size();
    for (int i=0; i<num2; i++)
      hyperplane[sparse_points[i2].id[i]] += t2*sparse_points[i2].value[i];            
  }

  // update error cache using new alphas
  for (int i=0; i<num_examples; i++)
    if (alpha1_old > 0 && alpha1_old < slack_penalty)
      error_cache[i] += t1*(this->*kernel_function)(i1, i) + t2*(this->*kernel_function)(i2, i) - delta_threshold;
  error_cache[i1] = 0;
  error_cache[i2] = 0;

  alphas[i1] = alpha1_new;
  alphas[i2] = alpha2_new;

  return 1;
}

void svm_train::train() {
  int numChanged = 0;
  bool examineAll = true;
    
  while (numChanged > 0 || examineAll) {
    numChanged = 0;
    if (examineAll) 
      for(int k=0; k<num_examples; k++)
	numChanged += examine_example(k);
    else
      for(int k=0; k<num_examples; k++)
	if (alphas[k] != 0 && alphas[k] != slack_penalty)
	  numChanged += examine_example(k);
    if (examineAll)
      examineAll = false;
    else if (numChanged == 0)
      examineAll = true;  
  }
  // write trained model to file
  write_svm();
}

void svm_train::write_svm() {
  ofstream svm_file(model_file);
  svm_file << num_features << endl;
  svm_file << is_linear_kernel << endl;
  svm_file << threshold << endl;
  // if (is_linear_kernel)
    for (int i=0; i<num_features; i++)
      svm_file << hyperplane[i] << endl;
    svm_file << "hyperplane ended"<<endl;
    // else {
    svm_file << two_sigma_squared << endl;
    int num_sv = 0;
    for (int i=0; i<num_examples; i++)
      if (alphas[i]>0)
	num_sv++;
    cerr<<"num sv="<<num_sv<<endl;
    svm_file << num_sv << endl;
    for(int i=0; i<num_examples; i++)
      if (alphas[i] > 0)
	svm_file << alphas[i] << endl;
    for(int i=0; i<num_examples; i++) {
      if (alphas[i] > 0) {
	svm_file << labels[i] << " ";
	for (unsigned int j=0; j<sparse_points[i].id.size(); i++)
	  svm_file << sparse_points[i].id[j] << ":" <<sparse_points[i].value[j]<<" ";
	svm_file<<endl;
      } 
    }
    //}
}

int main(int argc, char** argv) {  
  char* data_file_name;
  char* model_file_name = "svm.model";
  
  float slack_penalty = 0.05; // C
  float tolerance = 0.001; 
  float epsilon = 0.01;  
  float two_sigma_squared = 2;

  bool is_linear_kernel = false;

  // read parameters
  extern char* optarg;
  extern int optind;
  int c;
  int err_flag = 0;
  bool data_file_specified = false;
  while ((c=getopt(argc, argv, "d:c:t:e:s:f:m:rl")) != -1) {
    switch (c) {   
    case 'c':
      slack_penalty = atof(optarg);
      break;
    case 't':
      tolerance = atof(optarg);
      break;
    case 'e':
      epsilon = atof(optarg);
      break;
    case 's':
      two_sigma_squared = atof(optarg);
      break;
    case 'f':
      data_file_specified = true;
      data_file_name = optarg;
      break;
    case 'm':
      model_file_name = optarg;
      break;
    case 'r':
      break;
    case 'l':
      is_linear_kernel = true;
      break;    
    case '?':
      err_flag++;
    }    
  }
  if (!data_file_specified) {
    cout<<"Data file must be specified"<<endl;
    err_flag++;
  }    
  if (err_flag || optind < argc) {
    cerr<< "usage: "<<argv[0]<<" "<<endl<<
      "-f data_file_name"<<endl<<
      "-m svm_file_name"<<endl<<
      "-o output_file_name"<<endl<<
      "-c slack_penalty (i.e. C)"<<endl<<
      "-t tolerance"<<endl<<
      "-e epsilon"<<endl<<
      "-l (if linear kernel to be used)"<<endl<<
      "-p two_sigma_squared (if non-linear kernel being used)"<<endl<<
      "-r random_seed"<<endl;
      exit(2);
  }
  cout<<"Reading data from "<<data_file_name<<
    ", writing model file "<<model_file_name<<
    ", using C="<<slack_penalty<<
    ", tolerance="<<tolerance<<
    ", epsilon="<<epsilon<<endl; 
  svm_train svm(data_file_name, 
		    model_file_name,
		    slack_penalty, 
		    tolerance, 
		    epsilon, 
		    two_sigma_squared, 
		    is_linear_kernel);
  svm.train();
  return 0;
}
