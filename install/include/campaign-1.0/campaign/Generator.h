#include <iostream>

#ifndef __GENERATOR_H_
#define __GENERATOR_H_

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

#endif
