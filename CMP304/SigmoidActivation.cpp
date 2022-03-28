#include "SigmoidActivation.h"

SigmoidActivation::SigmoidActivation()
{
}

SigmoidActivation::SigmoidActivation(unsigned int inputSize)
{
	InputSize = inputSize;
}

SigmoidActivation::~SigmoidActivation()
{
}

void SigmoidActivation::Forward()
{
    //make sure the batch fits the input size
    if (Input.size() % InputSize != 0)
    {
        printf("SIGMOID ACTIVATION ERROR! Input neurons do not match the given Input Size.\n");
        printf("Input neurons in network: %zu Given Input Size: %u \n", Input.size(), InputSize);
    }
    else
    {
        Output.resize(Input.size());
        for (unsigned int i = 0; i < Input.size(); i++)
        {
            Output[i] = 1.0f / (1 + exp(-Input[i]));  //sigmoid equation 
        }
    }
}

void SigmoidActivation::CalcDeltas(std::vector<float> nextLayerDeltas)
{
    NextLayerDeltas = nextLayerDeltas;
    LayerDeltas.clear();
    LayerDeltas.resize(nextLayerDeltas.size());

    for (unsigned int i = 0; i < nextLayerDeltas.size(); i++)
    {
        LayerDeltas[i] = nextLayerDeltas[i] * Output[i] * (1.0f - Output[i]);
    }
}

void SigmoidActivation::UpdateParams(float learningRate)
{
}
