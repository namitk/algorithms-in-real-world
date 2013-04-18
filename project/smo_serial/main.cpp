#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include "smo.h"

using namespace std;
int main(int argc, char** argv)
{
    parser(argc,argv);

    setupVars();
    dualityGap = computeDualityGap();
    bLowUp();
    int numChanged = 1;

    while(dualityGap>tolerance*fabs(dual))
    {
        numChanged = takeStep();
        if(!numChanged)
            break;
        update2Alphas(newA1,iUp,X1.label);
        update2Alphas(newA2,iLow,X2.label);

        for(int i=0;i<(int)allExamples.size();i++)
        {
            fcache[i]+= (newA1-oldA1)*X1.label*kernel(X1,allExamples[i]);
            fcache[i]+= (newA2-oldA2)*X2.label*kernel(X2,allExamples[i]);
        }
        bLowUp();
        bAvg = (bLow+bUp)/2;
        dualityGap = computeDualityGap();
    }
    // Why again?
    bAvg = (bLow+bUp)/2;
    dualityGap = computeDualityGap();
    primal = dual + dualityGap;

    gatherAlphas();
    return 0;
}

void setupVars()
{
    numFeatures = read_data(data_file_name);
    alpha = new double[allExamples.size()];
    fcache = new double[allExamples.size()];
    I = new int[allExamples.size()];
    for(int i=0;i<(int)allExamples.size();i++)
    {
        fcache[i] = -allExamples[i].label;
        alpha[i] = 0;
        if(allExamples[i].label>0)
            I[i] = 1;
        else// if(allExamples[i].label<0)
            I[i] = 4;
    }
    dual = 0;
}


void getAFX(double &A,double &F,Example &X,int i)
{
    A = alpha[i];
    F = fcache[i];
    X = allExamples[i];
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

    Pair *sv = new Pair[allExamples.size()];
    for(int i=0;i<numExamples;i++)
    {
        sv[i].b = alpha[i];
        if(alpha[i]>ZERO)
            sv[i].rank = 1;
        else
            sv[i].rank = 0;
    }
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

double computeDualityGap()
{
    double myGap = 0;
    for(int i=0;i<(int)allExamples.size();i++)
    {
        myGap += slack_penalty * max(0.0,allExamples[i].label*(bAvg-fcache[i]));
        if(I[i]==0||I[i]==2||I[i]==3)
            myGap += alpha[i]*allExamples[i].label*fcache[i];
    }
    return myGap;
}

void bLowUp()
{
    iLow = iUp = -1;
    bLow = -INF;
    bUp = INF;
    for(int i=0;i<(int)allExamples.size();i++)
    {
        if(I[i]==0||I[i]==3||I[i]==4)
            if(fcache[i]>bLow)
                bLow = fcache[i],iLow = i;
        if(I[i]==0||I[i]==1||I[i]==2)
            if(fcache[i]<bUp)
                bUp = fcache[i],iUp = i;
    }
}
int takeStep()
{
    if(fabs(bUp-bLow)<ZERO)
        return 0;
    if(iUp==iLow)
        return 0;
    getAFX(oldA1,F1,X1,iUp);
    getAFX(oldA2,F2,X2,iLow);

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
