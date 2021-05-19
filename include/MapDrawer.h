#include"Map.h"
#include<iostream>
#include<string>
#include<pangolin/pangolin.h>

#ifndef MAPDRAWER_H
#define MAPDRAWER_H
namespace ORB_SLAM2
{
 using namespace std;   
class MapDrawer
{
private:
    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;
    cv::Mat mCameraPose;
    std::mutex mMutexCamera;
public:
    MapDrawer(Map* pMap, const string &strSettingPath);
    Map* mpMap;
    void DrawMapPoints();
    void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph);
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
    

 
};  
}
#endif