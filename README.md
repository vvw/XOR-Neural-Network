# XOR-Neural-Network

#

Basic MLP machine learning algorithm with back propagation written in C++.

Based on the <a href="https://vimeo.com/19569529">video tutorial</a> by David Miller.

#

# How to Operate

#

1. Compile the <a href="https://github.com/Isaacdelly/XOR-Neural-Network/blob/master/trainingDataGenerator.cpp">trainingDataGenerator.cpp</a>. This program will create a text file `trainingData.txt` that includes a list of inputs and the correct output that the computer will attempt to calculate. The generator could be modified to include different amounts of data.

2. Compile the <a href="https://github.com/Isaacdelly/XOR-Neural-Network/blob/master/neuralNetwork.cpp">neuralNetwork.cpp</a>. This program uses the `trainingData.txt` and uses it in a machine learning algorithm. The output of this program will be in `output.txt`. Using the predetermined settings of a 2-4-1 layer scheme, the machine will accurately calculate a XOR scenario within 99.98% accuracy by the end of 10,000 calculations.

#
