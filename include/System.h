#ifndef SYSTEM_H
#define SYSTEM_H

#include<iostream>
#include<string>
#include "Tracking.h"
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include "Viewer.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "MapDrawer.h"
#include <mutex>
#include<thread>

using namespace std;

namespace ORB_SLAM2
{

class Tracking;
class Viewer;
class FrameDrawer;
class Map;
class System
{
public:
    //Input sensor
    //这个枚举类型用于表示本系统所使用的传感器的类型
    enum eSensor{
        MONOCULAR=0,
        STEREO=1,
        RGBD=2

    };
public:
        //构造函数，用于初始化整个系统
        System(const string &strVocFile, //指定ORB字典文件
               const string &strSettingFile, //指定配置文件的路径
               const eSensor sensor,     //指定所使用的传感器的类型
               const bool bUseViewer = true); //指定是否使用可视化界面TODO



        cv::Mat TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap,const cv::Mat &mask, const double &timestamp);
        void ActivateLocalizationMode();
        void DeactivateLocalizationMode();
        // Reset the system (clear map)
        void Reset();
        void Shutdown();
private:
    eSensor mSensor;
    ORBVocabulary* mpVocabulary;
    KeyFrameDatabase* mpKeyFrameDatabase;
    Map* mpMap;
    Tracking* mpTracker;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;
    Viewer* mpViewer;
    std::thread* mptViewer;
    std::mutex mMutexReset;
    bool mbReset;
    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;
    bool mbDeactivateLocalizationMode;
    // Tracking state
    int mTrackingState;
    std::vector<MapPoint*> mTrackedMapPoints;
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
    std::mutex mMutexState;
};


} // namespace ORB_SLAM2
#endif