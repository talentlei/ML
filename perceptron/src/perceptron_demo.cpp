#include <string>
#include <iostream>
#include <vector>
#include "perceptron.h"

using namespace std;
int main()
{
	string trainfile = "../data/spambase.data";
	// string trainfile = "../data/simple.data";
	string model = "../bin/model";
	vector<vector<double> > trainSet;
	vector<int> labels;
	


	perceptron per;
	per.loadTrainData(trainfile,",",trainSet,labels);

	if(trainSet.size() == 0)
	{
		cerr << "no train data !" <<endl;
		return 1;
	}
	int dim = trainSet[0].size();
	double lrate = 0.1;
	vector<double> w(dim,0);
	double b = 0;
	int maxiter  = 10000;
	double delt  = 0.0001;


	per.train(trainSet, labels, model, lrate, w , b, maxiter,delt );
}