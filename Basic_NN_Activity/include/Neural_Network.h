#ifndef NEURAL_NETWORK_H
#define LEARNING_RATE 0.01

#include "vector"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include "math.h"
using namespace std;

// The Node class is used to replicate a Node in a neural network.
// It performs nonlnearity calculations and stores different values
// relevant to the weights and errors.

class Node
{
    public:

        // Constructor containing some basic values.
        Node(int out, int bias){

            output = out;
            input_bias = bias;

        }

        double input_bias;

        // Calculates the output of the node to be used in the next layer.
        void CalculateOutput(vector<double> weights, vector<double> un_weighted_inputs){

            // Multiplying all of the outputs of the nodes in the previous layer by the weights connecting the two.
            // The output is currently a cumulative sum of all of these values and then they are passed through the
            // Activation function.
            for(int i = 0; i < weights.size(); i++){
                output += weights[i]*un_weighted_inputs[i];
            }

            // Adding the bias of the Node.
            output+=input_bias;

            // Passing the output to the activation function to introduce nonlinearity.
            Activation();

            // Calculating the derivative for use later in the process.
            ActivationDerivative();

        }

        // This activation function is used to introduce nonlinearity to the system.
        void Activation(){

            // This is the sigmoid activation function.
            output = 1/(1+exp(-output));

            // It's good practice to round the sigmoid output to the nearest value since it is asymptotic.
            if(output > 0.9){
                output = 1.0;
            }else if(output < 0.1){
                output = 0.0;
            }

        }

        // This function is the exact derivative of the sigmoid function.
        void ActivationDerivative(){

                derivative = output*(1-output);

        }

        double output;
        double error;
        double derivative;

    protected:

    private:

};


// This is the overarching class of the neural network. Connects all of the nodes together and contains the algorithms
// for using the network.
class Neural_Network
{
    public:
        Neural_Network();
        virtual ~Neural_Network();

        Neural_Network(int layer_size, int input_size, int output_size)
        {
            input_number = input_size;
            hidden_number = layer_size;
            output_number = output_size;
            hidden_nodes.clear();
            for(int i = 0; i < layer_size; i++){
                hidden_nodes.push_back(Node(0.0, 0.0));
            }

            output_nodes.clear();
            for(int i = 0; i < output_size; i++){
                output_nodes.push_back(Node(0.0, 0.0));
            }

            srand(time(0));

            /*For each hidden layer node, make a new set of weights*/
            for(int i = 0; i < layer_size; i++){
                vector<double> row;
                hidden_input_weights.push_back(row);
                /* Make a new weight for every input node for every input*/
                for(int j = 0; j < input_size; j++){
                    hidden_input_weights[i].push_back(((double)rand())/RAND_MAX - 0.5);
                }
            }

            /*For each output layer node, make a new set of weights*/
            for(int i = 0; i < output_size; i++){
                vector<double> row;
                output_hidden_weights.push_back(row);
                /* Make a new weight for every hidden node for every input*/
                for(int j = 0; j < layer_size; j++){
                    output_hidden_weights[i].push_back(((double)rand())/RAND_MAX - 0.5);
                }
            }

        }

        vector<Node> hidden_nodes;
        vector<Node> output_nodes;
        vector<vector<double> > hidden_input_weights;
        vector<vector<double> > output_hidden_weights;
        vector<double> inputs;
        int input_number;
        int hidden_number;
        int output_number;
        vector<double> hidden_ys;

        void SetInputs(vector<double> global_inputs){

            inputs.clear();

            for(int i = 0; i < global_inputs.size(); i++){
                inputs.push_back(global_inputs[i]);
            }

        }


        void ForwardPropagation(){

            hidden_ys.clear();
            for(int i = 0; i < hidden_number; i++){
                hidden_nodes[i].CalculateOutput(hidden_input_weights[i], inputs);
                hidden_ys.push_back(hidden_nodes[i].output);
            }

            for(int i = 0; i < output_number; i++){
                output_nodes[i].CalculateOutput(output_hidden_weights[i], hidden_ys);
            }

        }

        void OutputErrorCalculation(double output){

            output_nodes[0].error = output - output_nodes[0].output;

        }

        void NodeErrorCalculation(){

            for(int i = 0; i < hidden_number; i++){
                hidden_nodes[i].error = output_nodes[0].error * output_hidden_weights[0][i];
            }
        }

        void UpdateWeights(){
            for(int i = 0; i < hidden_number; i++){
                    for(int j = 0; j < input_number; j++){
                        hidden_input_weights[i][j] += 1 * hidden_nodes[i].error * hidden_nodes[i].derivative * inputs[j];
                    }
                    hidden_nodes[i].input_bias += 1 * hidden_nodes[i].error * hidden_nodes[i].derivative;
            }
            for(int i = 0; i < output_number; i++){
                    for(int j = 0; j < hidden_number; j++){
                        output_hidden_weights[i][j] += 1 * output_nodes[i].error * output_nodes[i].derivative * hidden_nodes[j].output;
                    }
                    output_nodes[i].input_bias += 1 * output_nodes[i].error * output_nodes[i].derivative;
            }

            cout << "Inputs: " << inputs[0] << " " << inputs[1] << endl;
            cout << "Output: " << output_nodes[0].output << endl;


}



    protected:

    private:

};

#endif // NEURAL_NETWORK_H
