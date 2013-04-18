#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <mpi.h>

#include "smo.h"

using namespace std;
int main(int argc, char** argv)
{

    MPI_Init(NULL,NULL);
    parser(argc,argv);

    MPI_Comm_size(MPI_COMM_WORLD, &numProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    setupVars();
    refreshDualityGap();
    chooseAlphas();
    int numChanged = 1;

    while(dualityGap>tolerance*fabs(dual))
    {
        if(myRank==0)
            numChanged = takeStep();
        MPI_Bcast(&numChanged,1,MPI_INT,0,MPI_COMM_WORLD);
        if(!numChanged)
            break;
        MPI_Bcast(&newA1,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Bcast(&newA2,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Bcast(&dual,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        if(myRank==Z1)
            update2Alphas(newA1,iUpLocal,X1.label);
        if(myRank==Z2)
            update2Alphas(newA2,iLowLocal,X2.label);

        for(int i=0;i<(int)localExamples.size();i++)
        {
            fcache[i]+= (newA1-oldA1)*X1.label*kernel(X1,localExamples[i]);
            fcache[i]+= (newA2-oldA2)*X2.label*kernel(X2,localExamples[i]);
        }
        chooseAlphas();
        bAvg = (bLowGlobal+bUpGlobal)/2;
        refreshDualityGap();
    }
    // Why again?
    bAvg = (bLowGlobal+bUpGlobal)/2;
    refreshDualityGap();
    primal = dual + dualityGap;

    gatherAlphas();

    MPI_Finalize();
    return 0;
}

void setupVars()
{
    numFeatures = read_data(data_file_name);
    initLocalData(myRank);
    alpha = new double[localExamples.size()];
    fcache = new double[localExamples.size()];
    I = new int[localExamples.size()];
    for(int i=0;i<(int)localExamples.size();i++)
    {
        fcache[i] = -localExamples[i].label;
        alpha[i] = 0;
        if(localExamples[i].label>0)
            I[i] = 1;
        else// if(localExamples[i].label<0)
            I[i] = 4;
    }
    dual = 0;
}

void chooseAlphas()
{
    bLowUpLocal();

    Pair myUp(bUpLocal,myRank),out;
    MPI_Allreduce(&myUp,&out,1,MPI_DOUBLE_INT,MPI_MINLOC,MPI_COMM_WORLD);
    bUpGlobal = out.b;//useless ?
    Z1 = out.rank;

    Pair myLow(bLowLocal,myRank);
    MPI_Allreduce(&myLow,&out,1,MPI_DOUBLE_INT,MPI_MAXLOC,MPI_COMM_WORLD);
    bLowGlobal = out.b;//useless ?
    Z2 = out.rank;

    bCastAlphas(oldA1,F1,X1,iUpGlobal,iUpLocal,Z1);
    bCastAlphas(oldA2,F2,X2,iLowGlobal,iLowLocal,Z2);
}

void bCastAlphas(double &A,double &F,Example &X,int &iGlobal,int i,int bCaster)
{
    if(myRank==bCaster)
    {
        A = alpha[i];
        F = fcache[i];
        X = localExamples[i];
        iGlobal = i;
    }
    MPI_Bcast(&A,1,MPI_DOUBLE,bCaster,MPI_COMM_WORLD);
    MPI_Bcast(&F,1,MPI_DOUBLE,bCaster,MPI_COMM_WORLD);
    bCastExample(X,bCaster);
    MPI_Bcast(&iGlobal,1,MPI_INT,bCaster,MPI_COMM_WORLD);
}

void update2Alphas(double A,int i,int y)
{
    alpha[i] = A;
    if(y>0)
    {
        if(fabs(A-slack_penalty)<ZERO)
            I[i] = 3;
        else if(A<ZERO)
            I[i] = 1;
        else
            I[i] = 0;
    }
    else
    {
        if(fabs(A-slack_penalty)<ZERO)
            I[i] = 2;
        else if(A<ZERO)
            I[i] = 4;
        else
            I[i] = 0;
    }
}

void gatherAlphas()
{
    int numExamples = allExamples.size();
    int numLocal = localExamples.size();

    Pair *svLocal = new Pair[localExamples.size()];
    for(int i=0;i<numLocal;i++)
    {
        svLocal[i].b = alpha[i];
        if(alpha[i]>ZERO)
            svLocal[i].rank = 1;
        else
            svLocal[i].rank = 0;
    }

    Pair *sv;
    int *rcvCounts,*offsets;
    if(myRank==0)
    {
        sv = new Pair[numExamples];
        rcvCounts = new int[numProcessors];
        offsets = new int[numProcessors];
        for(int p=0;p<numProcessors;p++)
        {
            offsets[p] = (p*numExamples)/numProcessors;
            rcvCounts[p] = ((p+1)*numExamples/numProcessors) - offsets[p];
        }
    }
    MPI_Gatherv(svLocal,numLocal,MPI_DOUBLE_INT,sv,rcvCounts,offsets,MPI_DOUBLE_INT,0,MPI_COMM_WORLD);
    if(myRank==0)
        writeSVM(sv);
}

void writeSVM(Pair *sv)
{
    int numExamples = allExamples.size();
    int numSV = 0;
    for(int i=0;i<numExamples;i++)
        if(sv[i].b>ZERO)
            numSV++;
    cerr<<"["<<model_file_name<<"]"<<endl;
    ofstream svm(model_file_name);
    svm<<numFeatures<<endl<<isLinearKernel<<endl;
    svm<<bAvg<<endl;
    svm<<numSV<<endl;
    cerr<<"numSV = "<<numSV<<endl;
    for(int i=0;i<numExamples;i++)
        if(sv[i].b>ZERO)
            svm<<sv[i].b<<endl;
    for(int i=0;i<numExamples;i++)
    {
        if(sv[i].b>ZERO)
        {
            Example e = allExamples[i];
            svm<<e.label;
            for(int j=0;j<(int)e.id.size();j++)
                svm<<" "<<e.id[j]<<":"<<e.value[j];
            svm<<endl;
        }
    }
    svm.close();
}

void initLocalData(int rank)
{
    int total = allExamples.size();
    int start = (rank*total)/numProcessors;
    int end = ((rank+1)*total)/numProcessors;
    for(int i=start;i<end;i++)
        localExamples.push_back(allExamples[i]);
}

void refreshDualityGap()
{
    double myGap = computeDualityGap();
    MPI_Allreduce(&myGap,&dualityGap,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}

double computeDualityGap()
{
    double myGap = 0;
    for(int i=0;i<(int)localExamples.size();i++)
    {
        myGap += slack_penalty * max(0.0,localExamples[i].label*(bAvg-fcache[i]));
        if(I[i]==0||I[i]==2||I[i]==3)
            myGap += alpha[i]*localExamples[i].label*fcache[i];
    }
    return myGap;
}

void bLowUpLocal()
{
    iLowLocal = iUpLocal = -1;
    bLowLocal = -INF;
    bUpLocal = INF;
    for(int i=0;i<(int)localExamples.size();i++)
    {
        if(I[i]==0||I[i]==3||I[i]==4)
            if(fcache[i]>bLowLocal)
                bLowLocal = fcache[i],iLowLocal = i;
        if(I[i]==0||I[i]==1||I[i]==2)
            if(fcache[i]<bUpLocal)
                bUpLocal = fcache[i],iUpLocal = i;
    }
}

void bCastExample(Example& X,int bCaster)
{
    int numElements = 0;
    if(myRank==bCaster)
        numElements = X.id.size();
    MPI_Bcast(&numElements,1,MPI_INT,bCaster,MPI_COMM_WORLD);

    int *id = new int[numElements];
    double *val = new double[numElements];
    int label;
    if(myRank==bCaster)
    {
        label = X.label;
        for(int i=0;i<numElements;i++)
        {
            id[i] = X.id[i];
            val[i] = X.value[i];
        }
    }
    MPI_Bcast(&label,1,MPI_INT,bCaster,MPI_COMM_WORLD);
    MPI_Bcast(id,numElements,MPI_INT,bCaster,MPI_COMM_WORLD);
    MPI_Bcast(val,numElements,MPI_DOUBLE,bCaster,MPI_COMM_WORLD);
    X.label = label;
    X.id.clear();
    X.value.clear();
    X.id.insert(X.id.end(),id,id+numElements);
    X.value.insert(X.value.end(),val,val+numElements);
    delete[] id;
    delete[] val;
}

int takeStep()
{
    if(fabs(bUpGlobal-bLowGlobal)<ZERO)
        return 0;
    if(iUpGlobal==iLowGlobal&&Z1==Z2)
        return 0;
    int s = X1.label*X2.label;
    double gamma = oldA1 + s*oldA2;
    double L,H,K11,K12,K22,eta;
    if(s==1)
        L = max(0.0,gamma-slack_penalty), H = min(slack_penalty, gamma);
    else
        L = max(0.0,-gamma), H = slack_penalty - min(0.0, -gamma);
    if(H<=L)
        return 0;
    K11 = kernel(X1,X1);
    K22 = kernel(X2,X2);
    K12 = kernel(X1,X2);
    eta = 2*K12 - K11 - K22;
    if(eta<0)//epsilon*(K11+K22))
    {
        newA2 = oldA2 - X2.label*(F1-F2)/eta;
        newA2 = max(newA2,L);
        newA2 = min(newA2,H);
    }
    else
    {
        double slope = X2.label*(F1-F2);
        double change = slope * (H-L);
        if(fabs(change)>ZERO)
            if(slope>ZERO)
                newA2 = H;
            else
                newA2 = L;
        else
            newA2 = oldA2;
    }
    if(newA2 > slack_penalty*(1-epsilon))
        newA2 = slack_penalty;
    else if(newA2 < slack_penalty*epsilon)
        newA2 = 0;
    if(fabs(newA2-oldA2)<epsilon*(newA2+oldA2+epsilon))
        return 0;

    newA1 = gamma - s*newA2;
    if(newA1 > slack_penalty*(1-epsilon))
        newA1 = slack_penalty;
    else if(newA1 < slack_penalty*epsilon)
        newA1 = 0;

    double d = (newA1-oldA1)/X1.label;
    dual += -d*(F1-F2) + eta*d*d/2;
    return 1;
}

double dotProduct(const Example &u,const Example &v)
{
    double dot=0;
    for(int i=0,j=0;i<(int)u.id.size()&&j<(int)v.id.size();)
    {
        if(u.id[i]>v.id[j])
            j++;
        else if(u.id[i]<v.id[j])
            i++;
        else
        {
            dot += u.value[i]*v.value[j];
            i++;
            j++;
        }
    }
    return dot;
}

double kernel(const Example &u,const Example &v)
{
    if(isLinearKernel)
        return dotProduct(u,v);
    return 0;
}

int read_data(char* data_file)
{
    ifstream file(data_file);
    string line;
    int numFeatures = 0;
    if (!file.is_open())
        return 0;
    for (int numExamples = 0; file.good() ; numExamples++)
    {
        getline(file,line);
        istringstream ss(line);
        string token;
        getline(ss, token, ' ');
        Example ex;
        ex.label = atoi(token.c_str());
		if(ex.label>0)
			ex.label = 1;
		else
			ex.label = -1;
        while(getline(ss, token, ' ')) {
            int index; double val;
            sscanf(token.c_str(),"%d:%lf",&index,&val);
            numFeatures = max(numFeatures,index);
            ex.id.push_back(index);
            ex.value.push_back(val);
        }
        allExamples.push_back(ex);

    }
    return numFeatures+1;
}

void parser(int argc,char **argv)
{
    model_file_name = "svm.model";
    slack_penalty = 0.05; // C
    tolerance = 0.001;
    epsilon = 0.01;

    isLinearKernel = false;

    // read parameters
    extern char* optarg;
    extern int optind;
    int c;
    int err_flag = 0;
    bool data_file_specified = false;
    while ((c=getopt(argc, argv, "c:t:e:f:m:l")) != -1) {
        switch (c) {
            case 'c':
              sscanf(optarg,"%lf",&slack_penalty);
              break;
            case 't':
              sscanf(optarg,"%lf",&tolerance);
              break;
            case 'e':
              sscanf(optarg,"%lf",&epsilon);
              break;
            case 'f':
              data_file_specified = true;
              data_file_name = optarg;
              break;
            case 'm':
              model_file_name = optarg;
              break;
            case 'l':
              isLinearKernel = true;
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
          "-l (if linear kernel to be used)"<<endl;
          exit(2);
    }
}
