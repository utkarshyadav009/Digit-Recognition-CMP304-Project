#pragma once
#include "Layer.h"

class SigmoidActivation: public Layer
{
public:
	SigmoidActivation();
	SigmoidActivation(unsigned int inputSize);
	~SigmoidActivation();

	unsigned int InputSize;
	unsigned int OutputSize;

	void Forward();
	void CalcDeltas(std::vector<float> nextLayerDeltas);
	void UpdateParams(float learningRate);
};

