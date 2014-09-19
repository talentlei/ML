#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <sstream>
#include <fstream>
#include <math.h>
#include "perceptron.h"

using namespace std;
/**
 * 
 * [perceptron::train description]
 * @param trainSet [description]
 * @param labels   [1or -1 ]
 * @param model    [description]
 * @param lrate    [description]
 * @param w        [description]
 * @param b        [description]
 * @param maxiter  [description]
 * @param delt     [description]
 */

void perceptron::train(std::vector<std::vector<double> >& trainSet, std::vector<int>& labels, const std::string model, double lrate, 
	std::vector<double>& w , double& b, int maxiter, double delt )
{
	
	//验证数据的有效性
	if(trainSet.size() != labels.size())
	{
		cerr << "the size of trainSet and labels is not the same!!" << endl;
		return;
	}	
	for(vector<vector<double> >::const_iterator iter = trainSet.begin(); iter != trainSet.end(); iter++)
	{
		if((*iter).size() != w.size())
		{
			cerr << "the dim of  traindata is not the same with w" << endl;
			return ;
		}
	}
	if (maxiter <=0 )
	{
		cerr << "maxiter should larger than zero!" << endl;
		return;
	}
	if (delt <= 0)
	{
		cerr << "delt should larger than zero" << endl;
		return;
	}
	if (lrate <= 0)
	{
		cerr << "lrate should larger than zero" << endl;
		return;
	}

	// train
	bool haveBadCase = false;
	for(int iter = 0 ; iter <maxiter; ++iter)
	{
		haveBadCase = false;
		for (int i = 0; i < trainSet.size(); ++i)
		{
			// cout << i <<endl;
			vector<double> data = trainSet[i];
			int label = labels[i];
			bool bcase = isBadCase(data,w,label,b);
			if (bcase)
			{
				update(data,w,label,b,lrate);
				i--;
				haveBadCase = true;
			}
		}
 
		//normalization w b
		// normalize(w,b);

 		//no bad case 
		if(!haveBadCase )
			break;

		//cost less than delt
		double cost = 0.0;
		for (int i = 0; i < trainSet.size(); ++i)
		{
			vector<double> data = trainSet[i];
			double label = labels[i];
			double result = label *countMargin(data,w,b);
			if(result < 0)
				cost +=(-result);
		}
		if(cost < delt)
			break;
		
		cout << "-----------------------------------------------"<<endl;
		cout << "iter =" << iter << endl;
		cout << "cost =" << cost << endl; 
		print(w,b);
	}

	//print
	print(w,b);

}

void perceptron::print(vector<double> w, double b)
{
	stringstream ss;
	ss << "w=[";
	for (int i = 0; i < w.size()-1; ++i)
	{
		ss<<w[i]<<",";
	}
	ss<<w[w.size()-1]<<"]\n";
	
	string ws = ss.str();
	ss.clear();

	stringstream ss1;
	ss1 << "b="<<b<<"\n"<<endl;
	string bs = ss1.str();
	ss1.clear();

	cout << ws << endl;
	cout << bs << endl;
}
/**
 * [perceptron::countMargin description]
 * @param  data  [description]
 * @param  w     [description]
 * @param  label [description]
 * @param  b     [description]
 * @return       [description]
 */
double perceptron::countMargin(vector<double>& data, vector<double>& w, double b)
{
	//验证数据有效性
	if (data.size() != w.size())
	{
		cerr << "data dim is not equal with weight dim in function isBadCase" << endl;
		exit(EXIT_FAILURE);
	}

	//count 
	double result =0.0;
	for (int i = 0; i < data.size(); ++i)
	{
		result += data[i]*w[i];
	}
	result += b;

	return result;
}

bool perceptron::isBadCase(vector<double>& data, vector<double>& w, int label, double b)
{
	double result = label*countMargin(data,w,b);
	if (result <=0 )
		return true;
	else return false;

}

void perceptron::update(vector<double>& data, vector<double>& w, const double label, double & b,  const double lrate)
{
	//验证数据有效性
	if (data.size() != w.size())
	{
		cerr << "data dim is not equal with weight dim in function isBadCase" << endl;
		exit(EXIT_FAILURE);
	}

	//update w and b
	for (int i = 0; i < w.size(); ++i)
	{
		w[i] +=lrate*label*data[i];
	}

	b +=lrate*label;
	return;
}


//分割函数
std::vector<std::string> perceptron::split(const std::string& src, std::string separate_character)
{
	std::vector<std::string> strs;

	int separate_characterLen = separate_character.size();
	int lastPosition = 0,index = -1;
	while (-1 != (index = src.find(separate_character,lastPosition)))
	{
		strs.push_back(src.substr(lastPosition,index - lastPosition));
		lastPosition = index + separate_characterLen;
	}

	std::string lastString = src.substr(lastPosition);
	if (!lastString.empty())
		strs.push_back(lastString);

	return strs;
}


void perceptron::loadTrainData(const string datafile, const string  sep, vector<vector<double> >& trainSet , vector<int>& labels)
{
	ifstream in(datafile.c_str());
	if(!in)
	{
		cerr << "open file" << datafile<< " failed"<<endl;
		exit(EXIT_FAILURE);
	}

	string line ; 
	while(getline(in,line))
	{
		vector<string> array = split(line,sep);
		vector<double> temp ;
		try
		{
			for (int i = 0; i < array.size()-1; ++i)
			{					
				double num = atof(array[i].c_str());
				temp.push_back(num);
				
			}

			int label = atoi(array[array.size()-1].c_str());

			trainSet.push_back(temp);
			labels.push_back(label); 
		}
		catch(exception& e)
		{
			cerr << "line:"<< line <<" contain non-num items" << endl;
			cerr << e.what() << endl;
			exit(EXIT_FAILURE);
		}
	}

	in.close();
}

void perceptron::loadTestData(const string datafile, const string  sep, vector<vector<double> >& testSet )
{
	ifstream in(datafile.c_str());
	if(!in)
	{
		cerr << "open file" << datafile << " failed"<<endl;
		exit(EXIT_FAILURE);
	}

	string line ; 
	while(getline(in,line))
	{
		vector<string> array = split(line,sep);
		vector<double> temp ;
		try
		{
			for (int i = 0; i < array.size()-1; ++i)
			{					
				double num = atof(array[i].c_str());
				temp.push_back(num);
				
			}

			testSet.push_back(temp);
		}
		catch(exception& e)
		{
			cerr << "line:"<< line <<" contain non-num items" << endl;
			cerr << e.what() << endl;
			exit(EXIT_FAILURE);
		}
	}

	in.close();
}

void perceptron::test(std::vector<std::vector<double> >& testSet, std::vector<std::string>& plabels,const std::string model)
{

}


void perceptron::test(std::vector<std::vector<double> >& testSet, std::vector<int>& plabels,std::vector<double>& w , double& b )
{
	

	//验证数据有效性
	for (int i = 0; i < testSet.size(); ++i)
	{
		if(testSet[i].size() != w.size())
		{
			cerr << "the dim of  traindata is not the same with w" << endl;
			exit(EXIT_FAILURE) ;
		}
	}

	plabels.clear();
	for (int i = 0; i < testSet.size(); ++i)
	{
		double result = countMargin(testSet[i] , w, b);
		if(result > 0)
			plabels.push_back(1);	
		else plabels.push_back(-1);
	}
	return;
}
	
void perceptron::normalize(vector<double> & w, double & b)
{
	double result  = 0.0;
	for (int i = 0; i < w.size(); ++i)
	{
		result += w[i]*w[i];
	}

	result = sqrt(result);

	for (int i = 0; i < w.size(); ++i)
	{
		w[i] /= result;
	}
	b /=result;
	return;

}

	perceptron::perceptron()
	{}
 	perceptron::~perceptron()
 	{}


