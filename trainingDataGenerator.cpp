// Tool used to create a large amount of training data in learning stage

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>

using namespace std;

int main() {
	cout << "topology: 2 4 1" << endl; // Create 2 4 1 Level Layers
	ofstream out("trainingData.txt");
	for(int i = 10000; i >= 0; --i) {
        // Generate random training sets for XOR -- two inputs and one output
		int n1 = (int)(2.0 * rand() / double(RAND_MAX));
		int n2 = (int)(2.0 * rand() / double(RAND_MAX));
		int t = n1 ^ n2; // Should be 0 or 1
		out << "in: " << n1 << ".0 " << n2 << ".0 " << endl;
		out << "out: " << t << ".0" << endl; }
		out.close()
}
