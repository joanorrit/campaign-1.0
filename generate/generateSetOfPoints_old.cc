#include <iostream>
#include "time.h"

using namespace std;

#define RADIUS 205456

/**
 * Usage for an invalid input
 */
void usageInvalidInput() 
{
	printf("INVALID INPUT. You must enter N (points), K (clusters), D (dimensions) for generating the set of points\n");
	exit(0);
}

/**
 * Uasge for an N < K
 */
void usageLessPointsThanClusters()
{
	printf("INVALID INPUT N < K!!! You must enter more points thant clusters. N >> K\n");
	exit(0);
}

/**
 * Print all the points calculated for each cluster
 */
void printPoints(float * points, int totalPoints) 
{	
	for (int i = 0; i < totalPoints; i++)
	{
		cout << points[i] << endl;
	}
}

/**
 * Generates K centers
 */ 
void generateCenters(int K, float * centers)
{
	int centerModule = 1000000;
	for (int numCenter = 0; numCenter < K; numCenter++) {
		centers[numCenter] = (float) (rand() % centerModule) / ((float)1000);
		centerModule *= rand() % 8 + 1;
		if (!centerModule) centerModule += 1;
	}
}

/**
 * Generates all the points for each cluster. Stores the result at points
 */
void generateClusters(float * points, float * centers,  int totalPointsPerCluster, int K, int D)
{
	for (int numCluster = 0; numCluster < K; numCluster++) {
		for (int actualPointOfCluster = 0; actualPointOfCluster < totalPointsPerCluster; actualPointOfCluster++) {
			for (int dimensions = 0; dimensions < D; dimensions++) {
				int memory = (numCluster * totalPointsPerCluster + actualPointOfCluster) * D + dimensions;

				actualPointOfCluster % 2 == 0 ? 
					points[memory] = centers[numCluster] + (float) ((rand() % RADIUS)/(float)100) :
					points[memory] = centers[numCluster] - (float) ((rand() % RADIUS)/(float)100);
			}
		}
	}
}

/**
 * Generates a set of points with contiguous dimensions. 
 * A set of points with intercalate dimensions i required.
 */
void generateClustersContiguous(float * points, float * points_contiguous, int totalPoints, int dimensions)
{
	for (int actualPoint = 0; actualPoint < totalPoints/dimensions; actualPoint++) {
		int iniPositionPoint = actualPoint * dimensions;
		for (int dimension = 0; dimension < dimensions; dimension++){
			points_contiguous[actualPoint + dimension * (totalPoints/dimensions)] = points[iniPositionPoint + dimension];
		}
	}
}

int main(int argc, char *argv[])
{
	if (argc != 4)
		usageInvalidInput();

	int N = atoi(argv[1]);
	int K = atoi(argv[2]);
        int D = atoi(argv[3]);

	if (N < K)
		usageLessPointsThanClusters();

	int totalPointsPerCluster = N / K;
	int totalPoints = N*D;
	float * centers; centers = (float *) malloc(K * sizeof(float));
	float * points; points = (float *) malloc(totalPoints * sizeof(float));
	float * points_contiguous; points_contiguous = (float *) malloc(totalPoints * sizeof(float));
 
        generateCenters(K, centers);

	generateClusters(points, centers, totalPointsPerCluster, K, D); 

	generateClustersContiguous(points, points_contiguous, totalPoints, D);
	
	printPoints(points, totalPoints);

	//printPoints(points_contiguous, totalPoints);
} 
