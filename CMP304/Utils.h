#pragma once
#include <vector>

unsigned int MaxElement(std::vector<float> input)
{
    if (input.size() == 0)
        return -1;

    float max = input[0];
    int index = 0;
    for (unsigned int i = 1; i < input.size(); i++)
    {
        if (input[i] > max)
        {
            max = input[i];
            index = i;
        }
    }
    //printf("%u\n", index);
    return index;
}
std::vector<int> ProbabilityDistribution(std::vector<float> input)
{
    std::vector<int> Distribution;
    int Digits[10] = { 0 };
    for (unsigned int i = 0; i < input.size(); i++)
    {
        int p = input[i];
        switch (p)
        {
        case 0:
            Digits[0]++;
            break;
        case 1:
            Digits[1]++;
            break;
        case 2:
            Digits[2]++;
            break;
        case 3:
            Digits[3]++;
            break;
        case 4:
            Digits[4]++;
            break;
        case 5:
            Digits[5]++;
            break;
        case 6:
            Digits[6]++;
            break;
        case 7:
            Digits[7]++;
            break;
        case 8:
            Digits[0]++;
            break;
        case 9:
            Digits[9]++;
            break;
        default:
            break;
        }
    }
    for (int i = 0; i < 10; i++)
    {
        Distribution.push_back(Digits[i]);
    }
    return Distribution;
}
