# XOR-Neural-Network

#

Basic MLP machine learning algorithm with back propagation written in C++.

Based on the <a href="https://vimeo.com/19569529">video tutorial<\a> by David Miller.

#

# How to Operate

#

First compile the <a href="https://github.com/Isaacdelly/XOR-Neural-Network/blob/master/trainingDataGenerator.cpp">trainingDataGenerator.cpp</a>. This program will create a text file `trainingData.txt` that includes a list of inputs and the correct output that the computer will attempt to calculate. The generator could be modified to include different amounts of data.

Compile the <a href="https://github.com/Isaacdelly/XOR-Neural-Network/blob/master/neuralNetwork.cpp">neuralNetwork.cpp</a>. This program uses the `trainingData.txt` and uses it in a machine learning algorithm. The output of this program will be in `output.txt`. 
