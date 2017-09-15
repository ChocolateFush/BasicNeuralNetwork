#include <iostream>
// Number of input data points
#define INPUT_AMOUNT 2
// Amount of nodes in the hidden layers
#define HIDDEN_LAYER_SIZE 10
// Number of output data points
#define OUTPUT_AMOUNT 1
#include "Neural_Network.h"


using namespace std;

int main()
{
    Neural_Network nn = Neural_Network(HIDDEN_LAYER_SIZE, INPUT_AMOUNT, OUTPUT_AMOUNT);

    vector<double> micah;
    double micah_o;

    int iterator_ = 1;

    for(int i = 0; i < 2000; i++){

    micah.clear();
        if(iterator_ == 1){
            micah.push_back(0.0);
            micah.push_back(0.0);
            micah_o = 0.0;
            iterator_++;
        }else if(iterator_ == 2){
            micah.push_back(1.0);
            micah.push_back(0.0);
            micah_o = 1.0;
            iterator_++;
        }else if(iterator_ == 3){
            micah.push_back(0.0);
            micah.push_back(1.0);
            micah_o = 1.0;
            iterator_++;
        }else{
            micah.push_back(1.0);
            micah.push_back(1.0);

            micah_o = 0.0;
            iterator_ = 1;
        }

        nn.SetInputs(micah);

        nn.ForwardPropagation();

        nn.OutputErrorCalculation(micah_o);

        nn.NodeErrorCalculation();

        nn.UpdateWeights();

    }


//
//    for(int i = 0; i < 1000; i++){
//
//    micah[0] = 0.0;
//    micah[1] = 0.0;
//    micah_o = 0.0;
//
//    micah[0] = 1.0;
//    micah[1] = 0.0;
//    micah_o = 0.0;
//
//    micah[0] = 0.0;
//    micah[1] = 1.0;
//    micah_o = 0.0;
//
//    micah[0] = 1.0;
//    micah[1] = 1.0;
//    micah_o = 1.0;
//
//
//
//    }

    return 0;
}
