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
		void mallocMemoryForPoints();
		float * getIntercalatePoints();
		float * getContiguousPoints();
		int getN(int N);
		int getK(int K);
		int getD(int D);
		

	private:
		int N;
		int K;
		int D;
		int totalPoints;
		int totalPointsPerCluster;
		float * centers;
		float * points;
		float * pointsContiguous;
};

#endif
