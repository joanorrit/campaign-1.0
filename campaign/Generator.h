#ifdef HAVE_CONFIG_H
#include "../config.h"
#include "../campaign.h"
#else
// defined globally in campaign.h
// define distance metric, e.g. CAMPAIGN_DISTANCE_MANHATTAN, CAMPAIGN_DISTANCE_EUCLIDEAN_, etc.
#define CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED /** < Type of distance metric */
#define THREADSPERBLOCK 64 /** < Threads per block (tpb) */
#define FLOAT_TYPE float         /** < Precision of floating point numbers */

#undef _GLIBCXX_ATOMIC_BUILTINS

#include <iostream>
#include <cfloat>
#include "../util/dataio.h"
#include "../util/timing.h"
#include "../util/defaults.h"
#include "../util/metricsGPU.h"
#include "../util/gpudevices.h"
#endif

using namespace std;

class Generator {
	public:
		Generator();
		~Generator();
		void usageInvalidInput();
		void usageLessPointsThanClusters();
		void printPoints();
		void generateCenters();
		void generateClusters();
		void generateClustersContiguous();
		void setN(int N);
		void setK(int K);
		void setD(int D);	
		void activateContiguousAccess();
		void mallocMemoryForPoints();
		float * getIntercalatePoints();
		float * getContiguousPoints();
		float * generateClustersContiguousByIntercalatedPoints(float * points, int N, int D);
		float * generateClustersIntercalatedByContiguousPoints(float * pointsContiguous, int N, int D);
		int getN();
		int getK();
		int getD();
		bool contiguousAccessIsActivated();

	private:
		int N;
		int K;
		int D;
		int totalPoints;
		int totalPointsPerCluster;
		float * centers;
		float * points;
		float * pointsContiguous;
		bool accessContiguousActivated = false;
};
