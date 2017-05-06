#include "Generator.h"
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
        Generator generator;

	if (argc < 4)
		generator.usageInvalidInput();

	generator.setN(atoi(argv[1]));
	generator.setK(atoi(argv[2]));
        generator.setD(atoi(argv[3]));

	if (argc > 4) {
		for (int i = 4; i < argc; i++) {
			string arg = argv[i];
			if (arg == "-c" || arg == "--contiguous") {
				generator.activateContiguousAccess();
			}
		}
	}

	if (generator.getN() < generator.getK())
		generator.usageLessPointsThanClusters();

	generator.mallocMemoryForPoints();
 
        generator.generateCenters();

	generator.generateClusters(); 

	generator.generateClustersContiguous();
	
	generator.printPoints();
}

