#include "MNISTModel.h"

MNISTModel::MNISTModel()
{
}

MNISTModel::~MNISTModel()
{
	for (unsigned int i = 0; i < Layers.size(); i++)
	{
		delete Layers[i];
	}
}

std::vector<float> MNISTModel::Evaluate(std::vector<float> input)
{
	if (Layers.size() == 0)
	{
		printf("ERROR! Empty Model");
	}
	else
	{
		Layers[0]->Input = input;
		for (unsigned int i = 0; i < Layers.size(); i++)
		{
			Layers[i]->Forward();
			if (i < Layers.size() - 1)
			{
				Layers[i + 1]->Input = Layers[i]->Output;
			}
		}
		return Layers[Layers.size() - 1]->Output;
	}
	std::vector<float> Default;
	Default.push_back(-1);
	return Default;
}

float MNISTModel::Train(std::vector<float> input, std::vector<float> output, float learningRate)
{
	//Evaluate model 
	std::vector<float> modelOutput = Evaluate(input);

	//check output size 
	if (Layers[Layers.size() - 1]->Output.size() != output.size())
	{
		printf("ERROR! Output neurons do not match the given Output Size.\n");
		printf("Output neurons in network: %zu Given Output Size: %zu \n", Layers[Layers.size() - 1]->Output.size(), output.size());
		return -1;
	}

	//Reducing function calls 
	unsigned int oSize = output.size();

	//calculate the last layer deltas 
	std::vector<float> lastLayerDeltas;
	lastLayerDeltas.resize(modelOutput.size());
	float loss = 0.0f;

	for (unsigned int i = 0; i < lastLayerDeltas.size(); i++)
	{
		lastLayerDeltas[i] = modelOutput[i] - output[i % oSize];
		loss += lastLayerDeltas[i] * lastLayerDeltas[i] / 2;
	}

	//calculate layer deltas
	Layers[Layers.size() - 1]->CalcDeltas(lastLayerDeltas);
	for (int i = Layers.size() - 2; i >= 0; i--)
	{
		Layers[i]->CalcDeltas(Layers[i + 1]->LayerDeltas);  
	}

	//update model parameters
	for (unsigned int i = 0; i < Layers.size(); i++)
	{
		Layers[i]->UpdateParams(learningRate);
	}
	return loss;
}

void MNISTModel::AddNetwork(unsigned int inputSize, unsigned int outputSize)
{
	Network* network = new Network(inputSize, outputSize);
	Layer* layer = network;
	Layers.push_back(layer);
}

void MNISTModel::AddSoftmax(unsigned int inputSize)
{
	SoftmaxActivation* softmax = new SoftmaxActivation(inputSize);
	Layer* layer = softmax;
	Layers.push_back(layer);
}

void MNISTModel::AddSigmoid(unsigned int inputSize)
{
	SigmoidActivation* sigmoid = new SigmoidActivation(inputSize);
	Layer* layer = sigmoid;
	Layers.push_back(layer);
}

size_t MNISTModel::NumberOfLayers()
{
	return Layers.size();
}
