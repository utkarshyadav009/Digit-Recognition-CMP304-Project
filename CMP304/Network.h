#pragma once
#include "Layer.h"

class Network: public Layer 
{
public:
	Network();
	Network(unsigned int inputSize, unsigned int outputSize);
	~Network();
	
	unsigned int InputSize;
	unsigned int OutputSize;

	std::vector<float> Weight;
	std::vector<float> Bias;

	void Forward();
	void CalcDeltas(std::vector<float> nextLayerDeltas);
	void UpdateParams(float learningRate);
};

