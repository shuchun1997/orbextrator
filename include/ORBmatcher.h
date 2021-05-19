#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include"Frame.h"

namespace ORB_SLAM2
{
class ORBmatcher
{
public:
    ORBmatcher(float nnratio = 0.6, bool checkOri = true);
    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono);

public:
    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;


protected:

    float mfNNratio;
    float mbCheckOrientation;
    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

    
};

} //namespace ORB_SLAM2
#endif