#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H
#include<Tracking.h>
#include "Map.h"
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

using namespace std;
namespace ORB_SLAM2
{
class Frame;  //todo
class KeyFrame;
class Tracking;
class Viewer;  
class FrameDrawer
{

public:
    FrameDrawer(Map* pMap);
    //Update info from the last processed frame.
    void Update(Tracking *pTracker);
    // Draw last processed frame.
    cv::Mat DrawFrame();
protected:
    void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);
    // Info of the frame to be drawn
    int N;
    cv::Mat mIm;
    vector<cv::KeyPoint> mvCurrentKeys;
    vector<bool> mvbMap, mvbVO;
    bool mbOnlyTracking;
    vector<cv::KeyPoint> mvIniKeys;
    vector<int> mvIniMatches;
    int mnTracked, mnTrackedVO;
    Map* mpMap;
    int mState; 
    std::mutex mMutex;   
};

}
#endif