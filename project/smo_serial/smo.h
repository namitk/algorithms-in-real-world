#ifndef SMO_H_INCLUDED
#define SMO_H_INCLUDED

#include <vector>
#include <set>
#include <limits>
#include <float.h>

using namespace std;
/*** Custom Defined Structs ***/
struct Example {
    int label;
    vector<int> id;
    vector<double> value;
};

struct Pair {
    double b;
    int rank;
    Pair(double bb,int r)
    {
        b=bb;
        rank=r;
    }
    Pair()
    {
        Pair(0.0,0);
    }
};

/*** Global Variables ***/
const double ZERO = 1e-6;
const double INF = INFINITY;
int numFeatures;
vector<Example> allExamples;
double primal,dual,dualityGap;

double *alpha;
int *I;
double oldA1,oldA2,newA1,newA2,F1,F2;

Example X1,X2;
double *fcache;
int iLow,iUp;
double bAvg,bLow,bUp;

bool isLinearKernel;
char* data_file_name;
char* model_file_name;
double slack_penalty;
double tolerance;
double epsilon;
/*** Function Prototypes ***/
void parser(int argc,char **argv);
int read_data(char* data_file);
void setupVars();
double computeDualityGap();
void bLowUp();
void getAFX(double &A,double &F,Example &X,int i);
void update2Alphas(double A,int i,int y);
void gatherAlphas();
double dotProduct(const Example &,const Example &);
int takeStep();
double kernel(const Example &,const Example &);
void writeSVM(Pair*);
#endif // SMO_H_INCLUDED
