#ifndef TRACKING_H
#define TRACKING_H

#include"Viewer.h"
#include"System.h"
#include<string>
#include<iostream>
#include "ORBVocabulary.h"
#include "MapDrawer.h"
#include"KeyFrameDatabase.h"
#include"FrameDrawer.h"
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <mutex>
#include"Frame.h"
#include "Initializer.h"


namespace ORB_SLAM2
{
class System;
class FrameDrawer;
class Map;
class Viewer;
class Tracking
{
public:
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor);

    cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const cv::Mat &mask, const double &timestamp);     
    void InformOnlyTracking(const bool &flag);
    void SetViewer(Viewer* pViewer);
public:
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;
    int mSensor;
    Frame mInitialFrame;
    Frame mCurrentFrame;
    cv::Mat mImGray;
    cv::Mat mImRGB;
    cv::Mat mImDepth;
    bool mbOnlyTracking;
    std::vector<int> mvIniMatches;

    void Reset();
    




protected:



    void Track();
    // Map initialization for stereo and RGB-D
    void StereoInitialization();
    bool TrackWithMotionModel();
    void UpdateLastFrame();
    bool mbVO;
    float mDepthMapFactor;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;


    
    Initializer* mpInitializer;
    KeyFrame* mpReferenceKF;

    // System
    System* mpSystem;
    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    
    //Map
    Map* mpMap;
    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;


    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;
    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;


    
    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;
    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    Frame mLastFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;
    //Motion Model
    cv::Mat mVelocity;
    
};
}
#endif