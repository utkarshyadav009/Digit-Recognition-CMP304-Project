#include "SoftmaxActivation.h"

SoftmaxActivation::SoftmaxActivation()
{
	OutputSize = 0;
}

SoftmaxActivation::SoftmaxActivation(unsigned int inputSize)
{
	InputSize = inputSize;
}

SoftmaxActivation::~SoftmaxActivation()
{
}

void SoftmaxActivation::Forward()
{
    if (Input.size() % InputSize != 0)
    {
        printf("SOFTMAX ACTIVATION ERROR! Input neurons do not match the given Input Size.\n");
        printf("Input neurons in network: %zu Given Input Size: %u \n", Input.size(), InputSize);
    }
    else
    {
        Output.resize(Input.size());
        for (unsigned int i = 0; i < Input.size() / InputSize; i++)
        {
            float denominator = 0.0f;
            for (unsigned int j = 0; j < InputSize; j++)
            {
                int index = (j + i * InputSize);
                denominator += exp(Input[index]);
            }
            for (unsigned int j = 0; j < InputSize; j++)
            {
                int index = (j + i * InputSize);
                Output[index] = exp(Input[index]) / denominator;
            }
        }
    }
}

void SoftmaxActivation::CalcDeltas(std::vector<float> nextLayerDeltas)
{
    NextLayerDeltas = nextLayerDeltas;
    LayerDeltas.clear();
    LayerDeltas.resize(nextLayerDeltas.size());


    for (unsigned int i = 0; i < nextLayerDeltas.size(); i++)
    {
        LayerDeltas[i] = nextLayerDeltas[i] * Output[i] * (1.0f - Output[i]);
    }

    for (unsigned int i = 0; i < Input.size() / InputSize; i++)
    {
        for (unsigned int j = 0; j < InputSize; j++)
        {
            for (unsigned int k = 0; k < InputSize; k++)
            {
                if (j != k)
                    LayerDeltas[j + i * InputSize] += -1.0f * nextLayerDeltas[j + i * InputSize] * Output[j] * Output[k];
                else
                    LayerDeltas[j + i * InputSize] += nextLayerDeltas[j + i * InputSize] * Output[j] * (1.0f - Output[k]);
            }
        }
    }
}

void SoftmaxActivation::UpdateParams(float learningRate)
{
}
