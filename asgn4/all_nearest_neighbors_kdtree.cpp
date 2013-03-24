#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <queue>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

#define DIM 3
#define NUM_NEAREST_NEEDED 3
#define NUM_POINTS 1000000

struct point{
  double coordinates[DIM];
};

struct point_distance{
  point p;
  double distance;

  bool operator< (const point_distance& a) const {
    return distance<a.distance;
  }
};

priority_queue<point_distance> knearest;

class kdtree {
public:
  class compare_points{
    int axis;
  public:
    compare_points(int d){
      axis = d;
    }

    bool operator() (point a, point b){
      return a.coordinates[axis]<b.coordinates[axis];
    }
  };

  point location;
  int axis;
  kdtree* left_child;
  kdtree* right_child;  
  point_distance tmp;

  double sumaverage;  

  kdtree(point* point_list, int num_points, int depth) {
    sumaverage = 0.0;    
    if (num_points==0) {
      this->axis=-1;
      this->left_child = NULL;
      this->right_child = NULL;
    } else {      
      this->axis = depth % DIM;
      sort(point_list, point_list+num_points, compare_points(this->axis));
      int median = num_points/2;
      
      this->location = point_list[median];
      this->left_child = new kdtree(point_list, median, depth+1);
      this->right_child = new kdtree(point_list+median+1, num_points-median-1, depth+1);      
    }
  }

  void k_nearest_neighbors(point query) {
    if (this->axis==-1)
      return;
    else {     
      double cur_distance = distance(this->location, query);
      if (knearest.size() < NUM_NEAREST_NEEDED+1) {	
	tmp.p = this->location;
	tmp.distance = cur_distance;
	knearest.push(tmp);
      }      
      else if (cur_distance < knearest.top().distance) {
	knearest.pop();       
	tmp.p = this->location;
	tmp.distance = cur_distance;
	knearest.push(tmp);
      }
      
      kdtree* nc = near_child(query);
      nc->k_nearest_neighbors(query);

      double max_dist = knearest.top().distance;
      if (knearest.size()<NUM_NEAREST_NEEDED+1 || axis_distance(query) < max_dist) {
	kdtree* fc = far_child(query);
	fc->k_nearest_neighbors(query);
      }
      return;
    }
  }   

  void print_knearest(point query) {
    while(!knearest.empty()){
      knearest.pop();
    }    
    k_nearest_neighbors(query);
    int total = NUM_NEAREST_NEEDED;
    double sum = 0.0;
    while (total>0) {      
      sum += knearest.top().distance;
      knearest.pop();
      total--;
    }
    sumaverage += sum/NUM_NEAREST_NEEDED;
  }

  kdtree* near_child(point p) {
    if (p.coordinates[this->axis] < this->location.coordinates[this->axis])
      return this->left_child;
    else
      return this->right_child;
  }

  kdtree* far_child(point p) {
    if (p.coordinates[this->axis] < this->location.coordinates[this->axis])
      return this->right_child;
    else
      return this->left_child;
  }

  double axis_distance(point p) {
    double tmp = fabs(this->location.coordinates[this->axis] - p.coordinates[this->axis]);
    return tmp;
    // return tmp*tmp;
  }

  double distance(point a, point b) {
    double dist = 0.0;
    for (int i=0; i<DIM; i++) {
      double tmp = a.coordinates[i]-b.coordinates[i];
      dist += tmp*tmp;
    }
    return sqrt(dist);
  }
};

point parse_string_as_point(string input){
  point ret;
  istringstream iss(input);
  string n;
  int i=0;
  while(getline(iss, n, ',')) { 
    ret.coordinates[i] = atof(n.c_str());     
    i++;
  }  
  return ret;
}

int main(){
  string line;
  string input_file = "nkatariy.dat";
  ifstream file(input_file.c_str());
  if (file.is_open()) {
    getline(file, line);
    file.close();

    point* point_list = new point[NUM_POINTS];
    point* query_points = new point[NUM_POINTS];
    stringstream ss(line);
    string token;
    for(int i=0; i<NUM_POINTS; i++) {
      getline(ss, token, ';');
      point p = parse_string_as_point(token);
      point_list[i] = p;
      query_points[i] = p;
    }   
    // printf("Created point list\n");
    kdtree kdt(point_list, NUM_POINTS, 0);
    // printf("Built kdtree\n");
    for (int i=0; i<NUM_POINTS; i++){
      kdt.print_knearest(query_points[i]);
    }
    printf("Average distance: %lf\n", kdt.sumaverage);
  } else{
    cout<<"Could not open file"<<endl;
  }  
  
  return 0;
}
