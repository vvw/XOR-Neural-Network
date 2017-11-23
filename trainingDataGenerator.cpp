// MLP Neural Network
// Made By Isaac Delly
// https://github.com/Isaacdelly/XOR-Neural-Network

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>

using namespace std;

int main() {
	ofstream out("data.txt");
    out << "topology: 2 4 1" << endl;
	for(int i = 10000; i >= 0; --i) {
        // Generate random training sets for XOR -- two inputs and one output
		int n1 = (int)(2.0 * rand() / double(RAND_MAX));
		int n2 = (int)(2.0 * rand() / double(RAND_MAX));
		int t = n1 ^ n2;
		out << "in: " << n1 << ".0 " << n2 << ".0 " << endl;
		out << "out: " << t << ".0" << endl; }
		out.close();
		cout << "Training data created successfully" << endl;
		return 0;
}
