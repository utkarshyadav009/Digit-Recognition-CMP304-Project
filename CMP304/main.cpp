#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <time.h>
#include <cstdlib>

#include "MNISTModel.h"
#include "Utils.h"


std::vector<std::vector<float> > LoadMNIST();


int main()
{
    MNISTModel model;
    printf("Loading MNIST dataset in.....\n");

    std::vector<std::vector<float> > TrainingData = LoadMNIST();

    printf("Dataset Loaded...\n");

    //create labels
    std::vector<std::vector<float> > labels;
    for (int i = 0; i < 10; i++)
    {
        std::vector<float> label;
        label.resize(10);
        label[i] = 1.0f;
        labels.push_back(label);
    }

    //adding layers to the model 
    model.AddNetwork(784, 10);
    model.AddSoftmax(10);
    //model.AddSigmoid(10);
    const unsigned int InputSize = 784;
    const unsigned int NumClasses = 10;
    const unsigned int OutputSize = 10;
    const unsigned int NumEpoch = 10;
    const float LearningRate = 0.1f;
    const unsigned int NumExamples = TrainingData[0].size() / InputSize;

    printf("Training the Network...\n");

    for (unsigned int epoch = 0; epoch < NumEpoch; epoch++)
    {
        for (unsigned int i = 0; i < NumExamples; i++)
        {
            for (unsigned int j = 0; j < NumClasses; j++)
            {
                std::vector<float> input;
                input.reserve(InputSize);

                for (int k = i * InputSize; k < i * InputSize + InputSize; k++)
                    input.push_back(TrainingData[j][k]);

                float cost = model.Train(input, labels[j], LearningRate / pow(10, epoch / 10.0f));

                //train the model
                if (i % 100 == 0)
                    std::cout << "Epoch " << epoch + 1 << ", cost: " << cost << std::endl;
            }
        }

    }
    std::cout << "Testing trained model... " << std::endl;
    unsigned int numTests = 0;
    unsigned int numCorrect = 0;

    for (unsigned int i = 0; i < NumExamples; i++)
    {
        for (unsigned int j = 0; j < NumClasses; j++)
        {
            std::vector<float> input;
            input.reserve(InputSize);
            for (int k = i * InputSize; k < i * InputSize + InputSize; k++)
            {
                //std::cout << training_data[j][k] << std::endl;
                input.push_back(TrainingData[j][k]);
            }
            numTests++;

            if (MaxElement(model.Evaluate(input)) == j)
                numCorrect++;
        }
    }
    printf("Number of Tests: %u; Number of Correct Predictions: %u\n", numTests, numCorrect);
    float accuracy = ((float)numCorrect / numTests * 100.0f);
    printf("Accuracy: %f%%\n", accuracy);
    printf("Number of Layers in the NeuralNetwork model: %zu\n", model.NumberOfLayers());
    printf("\n\n\nClick Left Mouse button and Drag on the window to draw a digit between 0-9\nEnter to evaluate your input\nC to clear the input\nESC to quit\n");
    
    // Create the main window
    sf::RenderWindow window(sf::VideoMode(280, 280), "Handwriting Recognition");
    sf::VertexArray pointmap(sf::Points, 280 * 280);
    for (int i = 0; i < 280; i++)
    {
        for (int j = 0; j < 280; j++)
        {
            pointmap[i * 280 + j].position.x = j;
            pointmap[i * 280 + j].position.y = i;
            pointmap[i * 280 + j].color = sf::Color::Black;
        }
    }

    // Start the game loop
    while (window.isOpen())
    {

        // Process events
        sf::Event event;
        while (window.pollEvent(event))
        {
            // Close window: exit
            if (event.type == sf::Event::Closed) {
                window.close();
            }

            // Escape pressed: exit
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape) {
                window.close();
            }
        }

        //zoom into area that is left clicked
        if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
        {
            sf::Vector2i position = sf::Mouse::getPosition(window);
            if (position.x + position.y * 280 < 280 * 280 && position.x + position.y * 280 >= 0)
            {
                for (int i = -16; i < 17; i++)
                {
                    for (int j = -16; j < 17; j++)
                    {
                        if (position.x + i + (position.y + j) * 280 < 280 * 280 && position.x + i + (position.y + j) * 280 >= 0)
                        {
                            float distance_squared = i * i + j * j + 1;
                            sf::Color color(255 / distance_squared, 255 / distance_squared, 255 / distance_squared);
                            pointmap[position.x + i + (position.y + j) * 280].position.x = position.x + i;
                            pointmap[position.x + i + (position.y + j) * 280].position.y = position.y + j;
                            pointmap[position.x + i + (position.y + j) * 280].color += color;
                            pointmap[position.x + i + (position.y + j) * 280].color += color;
                            pointmap[position.x + i + (position.y + j) * 280].color += color;
                            pointmap[position.x + i + (position.y + j) * 280].color += color;
                            pointmap[position.x + i + (position.y + j) * 280].color += color;
                        }
                    }
                }
            }
        }
        //press c to clear the window 
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::C))
        {
            for (int i = 0; i < 280 * 280; i++)
                pointmap[i].color = sf::Color::Black;
        }

        //enter to send data for evaluation 
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Return))
        {
            std::vector<float> input;
            input.reserve(784);
            for (int k = 0; k < 28; k++)
            {
                for (int l = 0; l < 28; l++)
                {
                    float average = 0.0f;
                    for (int i = 0; i < 10; i++)
                        for (int j = 0; j < 10; j++)
                        {
                            float tempAverage = 0.0f;
                            tempAverage += pointmap[k * 10 * 280 + l * 10 + i * 280 + j].color.r;
                            tempAverage += pointmap[k * 10 * 280 + l * 10 + i * 280 + j].color.g;
                            tempAverage += pointmap[k * 10 * 280 + l * 10 + i * 280 + j].color.b;
                            tempAverage /= 3.0f;
                            average += tempAverage;
                        }
                    average /= 100.0f;
                    average /= 255.0f; //normalize
                    input.push_back(average);
                }
            }
            {   std::vector<float> output = model.Evaluate(input);
                int prediction = MaxElement(output);
                printf("This number is predicted to be a: %i \n\n", prediction);
            }
        }

        // Clear screen
        window.clear();
        window.draw(pointmap);

        // Update the window
        window.display();
    }
    return 0;
}