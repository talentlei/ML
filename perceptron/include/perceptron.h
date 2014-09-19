#ifndef  PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>
#include <string>	

/**
 * author : dreamercl
 * perceptron是感知机的一个c++实现
 */
class perceptron
 {
 public:
 	perceptron();
 	~perceptron();
	void train(std::vector<std::vector<double> >& trainSet, std::vector<int>& labels, const std::string model, double lrate, 
	std::vector<double>& w , double& b, int maxiter, double delt = 0.0000001 );
	void test(std::vector<std::vector<double> >& testSet, std::vector<std::string>& plabels,const std::string model);
	void test(std::vector<std::vector<double> >& testSet, std::vector<int>& plabels,std::vector<double>& w , double& b );
	void loadTrainData(const std::string datafile, const std::string  sep, std::vector<std::vector<double> >& trainSet , std::vector<int>& labels);
	void loadTestData(const std::string datafile, const std::string  sep, std::vector<std::vector<double> >& testSet );


private:
	double countMargin(std::vector<double>& data, std::vector<double>& w, double b);
	bool isBadCase(std::vector<double>& data, std::vector<double>& w, int label, double b);
	void update(std::vector<double>& data, std::vector<double>& w, const double label, double & b,  const double lrate);
	std::vector<std::string> split(const std::string& src, std::string separate_character);
	void normalize(std::vector<double> & w, double & b);
	void print(std::vector<double> w, double b );
	
 
 	/* data */
 }; 


#endif