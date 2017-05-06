/*
 * --------------------------------------------------------------------------- *
 *                                  CAMPAIGN                                   *
 * --------------------------------------------------------------------------- *
 * This is part of the CAMPAIGN data clustering library originating from       *
 * Simbios, the NIH National Center for Physics-Based Simulation of Biological *
 * Structures at Stanford, funded under the NIH Roadmap for Medical Research,  *
 * grant U54 GM072970 (See https://simtk.org), and the FEATURE Project at      *
 * Stanford, funded under the NIH grant LM05652                                *
 * (See http://feature.stanford.edu/index.php).                                *
 *                                                                             *
 * Portions copyright (c) 2010 Stanford University, Authors, and Contributors. *
 * Authors: Marc Sosnick                                                       *
 * Contributors: Kai J. Kolhoff, William Hsu                                   *
 *                                                                             *
 * This program is free software: you can redistribute it and/or modify it     *
 * under the terms of the GNU Lesser General Public License as published by    *
 * the Free Software Foundation, either version 3 of the License, or (at your  *
 * option) any later version.                                                  *
 *                                                                             *
 * This program is distributed in the hope that it will be useful, but WITHOUT *
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public        *
 * License for more details.                                                   *
 *                                                                             *
 * You should have received a copy of the GNU Lesser General Public License    *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.       *
 * --------------------------------------------------------------------------- *
 */

/* $Id: timing.cpp 158 2011-01-12 23:06:27Z msosnick $ */
/**
 * \file timing.cpp
 * \brief Named timer functionality
 * 
 * Provides multiple simultaneous named timers.
 * 
 * \author Author: Marc Sosnick, Contributors: Kai J. Kolhoff, William Hsu
 * \date 5/26/10
 * \version pre-0.1
 */

#include "timing.h"

Timing::Timing(){
}

Timing::~Timing(){
}


void Timing::init(string timerName){
    cudaEventCreate(&startMap[timerName]);
    cudaEventCreate(&stopMap[timerName]);  
    elapsedMap[timerName] = 0.0;
}

void Timing::reset(string timerName){
    elapsedMap[timerName] = 0.0;
}

void Timing::start(string timerName){
    cudaEventRecord(startMap[timerName], 0);
}


void Timing::stop(string timerName){
    cudaEventRecord(stopMap[timerName], 0);
    cudaEventSynchronize(stopMap[timerName]);
    float time;
    cudaEventElapsedTime(&time,startMap[timerName],stopMap[timerName]);
    elapsedMap[timerName] += time;

}


void Timing::report(){
    cudaEvent_t currentTime;
    cudaEventCreate(&currentTime);
    cudaEventRecord(currentTime,0);
    float timeMs;
    string status = "";
    
    cout << "Current Timings:" << endl;
    cout << setw(15) << "Timer" << setw(15) <<  "Time (ms)" << setw(15) << "Status" << endl;
    for( map<string, cudaEvent_t>::iterator it=startMap.begin(); it!=startMap.end() ; ++it){
        if(stopMap.find((*it).first) != stopMap.end()){
            //cudaEventElapsedTime(&timeMs, (*it).second, stopMap[(*it).first]);
            timeMs = elapsedMap[(*it).first];
            status="done";
        } else {
            cudaEventElapsedTime(&timeMs, (*it).second , currentTime);
            status="running";
        }
        
        cout << setw(15) << (*it).first << setw(15) << timeMs << setw(15) << status << endl;
    }
}

void Timing::report(string timerName){
    cudaEvent_t currentTime;
    cudaEventCreate(&currentTime);
    cudaEventRecord(currentTime,0);
    float timeMs;
    
    if(startMap.find(timerName) == startMap.end()){
        cout << "Timer \"" << timerName << "\" was never started." << endl;
        return;
    } else if(stopMap.find(timerName) == stopMap.end()){
        cudaEventElapsedTime(&timeMs, startMap[timerName], currentTime);
        cout << timerName << " = " << timeMs << " ms (running)" << endl;
        return;
    }
    //cudaEventElapsedTime(&timeMs, startMap[timerName], stopMap[timerName]);
    timeMs = elapsedMap[timerName];
    cout << timerName << " = " << timeMs << " ms" << endl;
}


double TimingCPU::getTimeInMicroseconds(void){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double t = tv.tv_sec*1e6 + tv.tv_usec;
    return t;
}

void TimingCPU::init(string timerName){
    elapsedMap[timerName] = 0.0;
}

void TimingCPU::start(string timerName){
    double t;
    t = getTimeInMicroseconds();
    startMap[timerName] = t;
}


void TimingCPU::stop(string timerName){
    double t;
    t = getTimeInMicroseconds();
    stopMap[timerName] = t;
    elapsedMap[timerName] = elapsedMap[timerName] + (stopMap[timerName] - startMap[timerName]);
}

void TimingCPU::reset(string timerName) {
    elapsedMap[timerName] =0.0;
}

//void TimingCPU::report(string timerName){
  //cout << "Elapsed time(" << timerName << "):  " << elapsedMap[timerName] << endl;
//}

void TimingCPU::report(){
    double timeMs;
    string status = "";
    
    cout << "Current Timings:" << endl;
    cout << setw(15) << "Timer" << setw(15) <<  "Time (ms)" << setw(15) << "Status" << endl;
    for( map<string, double>::iterator it=startMap.begin(); it!=startMap.end() ; ++it){
        if(stopMap.find((*it).first) != stopMap.end()){
            //timeMs = stopMap[(*it).first] - (*it).second;
            timeMs = elapsedMap[(*it).first];
            status="done";
        } else {
            timeMs = getTimeInMicroseconds() - (*it).second;
            status="running";
        }
        
        cout << setw(15) << (*it).first << setw(15) << (timeMs / 1000.0f) << setw(15) << status << endl;
    }
}

void TimingCPU::report(string timerName){
    double timeMs;
    
    if(startMap.find(timerName) == startMap.end()){
        cout << "Timer \"" << timerName << "\" was never started." << endl;
        return;
    } else if(stopMap.find(timerName) == stopMap.end()){
        timeMs = getTimeInMicroseconds() - startMap[timerName];
        timeMs = getTimeInMicroseconds() - startMap[timerName];
        cout << timerName << " = " << timeMs << " ms (running)" << endl;
        return;
    }
    //timeMs = stopMap[timerName] - startMap[timerName];
    timeMs = elapsedMap[timerName];
    cout << timerName << " = " << (timeMs / 1000.0f) << " ms" << endl;
}
