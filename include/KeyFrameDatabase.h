#ifndef KEYFRAMEDATABASE_H
#define KEYFRAMEDATABASE_H
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include <list>
namespace ORB_SLAM2
{
class KeyFrame;
class KeyFrameDatabase
{

public:
    KeyFrameDatabase(const ORBVocabulary &voc);


protected:
    const ORBVocabulary* mpVoc;
    std::vector<list<KeyFrame*> > mvInvertedFile;
};

}
#endif