#pragma once
#include <vector>
#include "Network.h"
#include "SigmoidActivation.h"
#include "SoftmaxActivation.h"

class MNISTModel
{
public:
	MNISTModel();
	~MNISTModel();
	
	std::vector<float> Evaluate(std::vector<float> input);
	
	float Train(std::vector<float> input, std::vector<float> output, float learningRate);

	void AddNetwork(unsigned int inputSize, unsigned int outputSize);
	void AddSoftmax(unsigned int inputSize);
	void AddSigmoid(unsigned int inputSize);

	std::vector<Layer* > Layers;
	size_t NumberOfLayers();
};

