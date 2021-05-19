#ifndef MAPPOINT_H
#define MAPPOINT_H
#include<mutex>
#include"Map.h"
#include"KeyFrame.h"
#include<opencv2/core/core.hpp>


namespace ORB_SLAM2
{
class KeyFrame;
class Map;
class Frame;

class MapPoint
{
public:
    MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
    MapPoint(const cv::Mat &Pos,  Map* pMap, Frame* pFrame, const int &idxF);
    cv::Mat GetWorldPos();
    bool isBad();
    int Observations();
    cv::Mat GetDescriptor();
    void AddObservation(KeyFrame* pKF,size_t idx);
    void ComputeDistinctiveDescriptors();
    void UpdateNormalAndDepth();

public:
    static long unsigned int nNextId;
    long unsigned int mnId;
    long int mnFirstKFid;
    long int mnFirstFrame;
    int nObs;
    //Variables used by the tracking 
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnLastFrameSeen;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;  

protected:
    cv::Mat mWorldPos;
     // Bad flag (we do not currently erase MapPoint from memory)
     bool mbBad;
     MapPoint* mpReplaced;

     std::mutex mMutexPos;
     std::mutex mMutexFeatures;
     // Keyframes observing the point and associated index in keyframe
     std::map<KeyFrame*,size_t> mObservations;

    // Mean viewing direction
     cv::Mat mNormalVector;
     // Best descriptor to fast matching
     cv::Mat mDescriptor;
    // Reference KeyFrame
     KeyFrame* mpRefKF;

    // Tracking counters
     int mnVisible;
     int mnFound;
    // Scale invariance distances
     float mfMinDistance;
     float mfMaxDistance;
     Map* mpMap;
};

}
#endif