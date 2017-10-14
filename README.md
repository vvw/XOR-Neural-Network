# XOR-Neural-Network

Basic MLP machine learning algorithm with back propagation written in C++.

Based on the <a href="https://vimeo.com/19569529" target="_blank">video tutorial</a> by David Miller.

#

# How to Operate

1. Compile the <a href="https://github.com/Isaacdelly/XOR-Neural-Network/blob/master/trainingDataGenerator.cpp">trainingDataGenerator.cpp</a>. This program will create a text file `trainingData.txt` that includes a list of 10,000 XOR scenarios along with a corresponding answer that the computer will attempt to calculate. The generator could be modified to include different amounts of data if desired.

2. Compile the <a href="https://github.com/Isaacdelly/XOR-Neural-Network/blob/master/neuralNetwork.cpp">neuralNetwork.cpp</a>. This program uses the `trainingData.txt` in a machine learning algorithm that will update connection weights in order to be closer to the correct output. The output of this program will be saved in `output.txt`. 

#

# Evaluations

Using the predetermined settings of a 2-4-1 layer scheme with 0.15 learning rate, the machine will accurately calculate a XOR scenario within 99.98% accuracy by the end of 10,000 calculations.

#
