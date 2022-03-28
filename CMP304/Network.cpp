#include "Network.h"

Network::Network()
{
}

Network::Network(unsigned int inputSize, unsigned int outputSize)
{
	InputSize = inputSize;
	OutputSize = outputSize;

	//setting random weights for initialisation 
	Weight.resize(InputSize * OutputSize);
	for (unsigned int i = 0; i < InputSize * OutputSize; i++)
	{
		Weight[i] = (rand() % 1000) / 100000.0f;
	}

	//setting random biases for the output 
	Bias.resize(OutputSize);
	for (unsigned int i = 0; i < OutputSize; i++)
	{
		Bias[i] = (rand() % 1000) / 100000.0f;
	}
}

Network::~Network()
{
}

void Network::Forward()
{
	if (Input.size() % InputSize != 0)
	{
		printf("NETWORK ERROR! Input neurons do not match the given Input Size.\n");
		printf("Input neurons in network: %zu Given Input Size: %u \n", Input.size(), InputSize);
	}
	else
	{
		//Resize the output to match the batch size
		Output.clear();
		Output.resize(Input.size() / InputSize * OutputSize);

		//reducing function calls 
		unsigned int iSize = Input.size();
		unsigned int oSize = Output.size();

		//weight calculations 
		for (unsigned int i = 0; i < iSize; i++)
		{
			for (unsigned int j = 0; j < OutputSize; j++)
			{
				int outputIndex = (j + (i / InputSize) * OutputSize);
				int weightIndex = (j + (i % InputSize) * OutputSize);
				
				Output[outputIndex] += Input[i] * Weight[weightIndex];
			}
		}

		//Bias calculations 
		for (unsigned int i = 0; i < oSize; i++)
		{
			Output[i] += Bias[i % OutputSize];
		}
	}
}

void Network::CalcDeltas(std::vector<float> nextLayerDeltas)
{
	//resize the deltas 
	NextLayerDeltas = nextLayerDeltas;
	LayerDeltas.clear();
	LayerDeltas.resize(nextLayerDeltas.size() / OutputSize * InputSize);

	//reducing function calls 
	unsigned int iSize = Input.size();

	//weight calculation 
	for (unsigned int i = 0; i < iSize; i++)
	{
		for (unsigned int j = 0; j < OutputSize; j++)
		{
			int deltaIndex  = (j + (i / InputSize) * OutputSize);
			int weightIndex = (j + (i % InputSize) * OutputSize);
			
			LayerDeltas[i] += NextLayerDeltas[deltaIndex] * Weight[weightIndex];
		}
	}
}

void Network::UpdateParams(float learningRate)
{
	//reducing function calls 
	unsigned int iSize = Input.size();
	unsigned int oSize = Output.size();
	unsigned int batchSize = iSize / InputSize;

	//Weight update
	for (unsigned int i = 0; i < iSize; i++)
	{
		for (unsigned int j = 0; j < OutputSize; j++)
		{
			int weightIndex = (j + (i % InputSize) * OutputSize);
			int deltaIndex = (j + (i / InputSize) * OutputSize);
			Weight[weightIndex] += -1.0f / batchSize * learningRate * Input[i] * NextLayerDeltas[deltaIndex];
		}
	}		

	//Bias update
	for (unsigned int i = 0; i < oSize; i++)
	{
		Bias[i % OutputSize] += -1.0f / batchSize * NextLayerDeltas[i] * learningRate;
	}
}
