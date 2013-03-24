#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <queue>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

#define DIM 3
#define NUM_NEAREST_NEEDED 3
#define NUM_POINTS 1000000
#define NUM_QUERIES 1000

double distance(double* a, double* b) {
  double dist = 0.0;
  for (int i=0; i<DIM; i++) {
    double tmp = a[i]-b[i];
    dist += tmp*tmp;
  }
  return sqrt(dist);
}

double getknn(double* p, double** point_list) {
  priority_queue<double> knearest;  
  for (int i=0; i<NUM_POINTS; i++) {
    if (knearest.size() < NUM_NEAREST_NEEDED+1) {
      knearest.push(distance(p, point_list[i]));
    } else {
      double d = distance(p, point_list[i]);
      if (d < knearest.top()) {
	knearest.pop();
	knearest.push(d);
      }
    }
  }
  double sum = 0.0;
  for(int i=0; i<NUM_NEAREST_NEEDED; i++) {
    sum += knearest.top();
    knearest.pop();
  }
  return sum/NUM_NEAREST_NEEDED;
}

double* parse_string_as_point(string input){
  double* ret = new double[DIM];
  istringstream iss(input);
  string n;
  int i=0;
  while(getline(iss, n, ',')) { 
    ret[i] = atof(n.c_str());     
    i++;
  }  
  return ret;
}

int main(){
  string line;
  string input_file = "namits.dat";
  ifstream file(input_file.c_str());
  double** point_list = new double*[NUM_POINTS];
  for (int i=0; i<NUM_POINTS; i++) {
    point_list[i] = new double[DIM];
  }
  if (file.is_open()) {
    getline(file, line);
    file.close();

    stringstream ss(line);
    string token;
    for(int i=0; i<NUM_POINTS; i++) {
      getline(ss, token, ';');      
      point_list[i] = parse_string_as_point(token);
    } 
    double sumaverage = 0.0;
    for (int i=0; i<NUM_QUERIES; i++){      
      sumaverage += getknn(point_list[i], point_list);
      // fprintf(stderr, "%d ", i);
    }
    printf("Average distance: %lf\n", sumaverage);
  } else{
    cout<<"Could not open file"<<endl;
  }  
  
  return 0;
}
