#include "Generator.h"

int main(int argc, char *argv[])
{
        Generator generator;

	if (argc != 4)
		generator.usageInvalidInput();

	generator.setN(atoi(argv[1]));
	generator.setK(atoi(argv[2]));
        generator.setD(atoi(argv[3]));

	if (N < K)
		usageLessPointsThanClusters();

	generator.mallocMemoryForPoints();
 
        generator.generateCenters();

	generator.generateClusters(); 

	generator.generateClustersContiguous();
	
	generator.printPoints();

	//printPoints(points_contiguous, totalPoints);
} 
