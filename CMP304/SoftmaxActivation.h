#pragma once
#include"Layer.h"

class SoftmaxActivation: public Layer
{	
public:
	SoftmaxActivation();
	SoftmaxActivation(unsigned int inputSize);
	~SoftmaxActivation();

	unsigned int InputSize;
	unsigned int OutputSize;

	void Forward();
	void CalcDeltas(std::vector<float> nextLayerDeltas);
	void UpdateParams(float learningRate);
};

