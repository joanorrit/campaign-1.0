#include "Generator.h"

using namespace std;

#define RADIUS 205456

Generator::Generator()
{
}

Generator::~Generator()
{
}

/**
 * Usage for an invalid input
 */
void Generator::usageInvalidInput() 
{
	printf("INVALID INPUT. You must enter N (points), K (clusters), D (dimensions) for generating the set of points\n");
	exit(0);
}

/**
 * Uasge for an N < K
 */
void Generator::usageLessPointsThanClusters()
{
	printf("INVALID INPUT N < K!!! You must enter more points thant clusters. N >> K\n");
	exit(0);
}

/**
 * Print all the points calculated for each cluster
 */
void Generator::printPoints() 
{	
	if (this->accessContiguousActivated) {
		for (int i = 0; i < this->totalPoints; i++)
		{	
			cout << this->pointsContiguous[i] << endl;
		}
	} else {
		for (int i = 0; i < this->totalPoints; i++)
		{		
			cout << this->points[i] << endl;
		}
	}
}

/**
 * Generates K centers
 */ 
void Generator::generateCenters()
{
	int centerModule = 1000000;
	for (int numCenter = 0; numCenter < this->K; numCenter++) {
		this->centers[numCenter] = (float) (rand() % centerModule) / ((float)1000);
		centerModule *= rand() % 8 + 1;
		if (!centerModule) centerModule += 1;
	}
}

/**
 * Generates all the points for each cluster. Stores the result at points
 */
void Generator::generateClusters()
{
	for (int numCluster = 0; numCluster < this->K; numCluster++) {
		for (int actualPointOfCluster = 0; actualPointOfCluster < this->totalPointsPerCluster; actualPointOfCluster++) {
			for (int dimensions = 0; dimensions < this->D; dimensions++) {
				int memory = (numCluster * this->totalPointsPerCluster + actualPointOfCluster) * this->D + dimensions;

				actualPointOfCluster % 2 == 0 ? 
					points[memory] = this->centers[numCluster] + (float) ((rand() % RADIUS)/(float)100) :
					points[memory] = this->centers[numCluster] - (float) ((rand() % RADIUS)/(float)100);
			}
		}
	}
}

/**
 * Generates a set of points with contiguous dimensions. 
 * A set of points with intercalate dimensions i required.
 */
void Generator::generateClustersContiguous()
{
	for (int actualPoint = 0; actualPoint < this->totalPoints/D; actualPoint++) {
		int iniPositionPoint = actualPoint * D;
		for (int dimension = 0; dimension < D; dimension++){
			this->pointsContiguous[actualPoint + dimension * (this->totalPoints/D)] = this->points[iniPositionPoint + dimension];
		}
	}
}

/**
 * Sets N value.
 */
void Generator::setN(int N)
{
	this->N = N;
}

/**
 * Sets K value.
 */
void Generator::setK(int K)
{
	this->K = K;
}

/**
 * Sets D value.
 */
void Generator::setD(int D)
{
	this->D = D;
}

/**
 * Mallocs all memory necessary for
 * generating the sets of points.
 */
void Generator::mallocMemoryForPoints() 
{
	this->totalPointsPerCluster = this->N / this->K;
	this->totalPoints = this->N * this->D;
	this->centers = (float *) malloc(this->K * sizeof(float));
	this->points = (float *) malloc(this->totalPoints * sizeof(float));
	this->pointsContiguous = (float *) malloc(this->totalPoints * sizeof(float));
}

/**
 * Gets the set of point in an intercalated structure
 */
float * Generator::getIntercalatePoints()
{
	return this->points;
}

/**
 * Gets the set of point in an contiguous structure
 */
float * Generator::getContiguousPoints()
{
	return this->pointsContiguous;
}

int Generator::getN()
{
	return this->N;
}

int Generator::getK()
{
	return this->K;
}

int Generator::getD()
{
	return this->D;
}

bool Generator::contiguousAccessIsActivated()
{
	return this->accessContiguousActivated;
}

void Generator::activateContiguousAccess()
{
	this->accessContiguousActivated = true;
}
