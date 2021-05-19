#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "/home/shuchun/SlamDemo/orbextrator/Thirdparty/DBoW2/DBoW2/BowVector.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM2
{
KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}    
}