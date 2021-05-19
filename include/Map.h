
#ifndef MAP_H
#define MAP_H
#include "MapPoint.h"
#include <set>
#include <mutex>
#include "KeyFrame.h"
namespace ORB_SLAM2
{
class KeyFrame;
class MapPoint;
class Map
{

public:
    Map();
    void AddKeyFrame(KeyFrame* pKF);
    void AddMapPoint(MapPoint* pMP);
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapPoint*> GetReferenceMapPoints();   
    std::vector<KeyFrame*> GetAllKeyFrames(); 
    std::mutex mMutexMapUpdate;
    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;
    
    long unsigned  KeyFramesInMap();
    long unsigned int MapPointsInMap();
protected:
    std::set<MapPoint*> mspMapPoints;
    std::set<KeyFrame*> mspKeyFrames;
    
    std::vector<MapPoint*> mvpReferenceMapPoints;
    long unsigned int mnMaxKFid;
    int mnBigChangeIdx;


    std::mutex mMutexMap;
    
};

    
    
}

#endif