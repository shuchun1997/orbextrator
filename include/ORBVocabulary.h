#ifndef ORBVOCABULARY_H
#define ORBVOCABULARY_H

#include "/home/shuchun/SlamDemo/orbextrator/Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h"
#include "/home/shuchun/SlamDemo/orbextrator/Thirdparty/DBoW2/DBoW2/FORB.h"



namespace ORB_SLAM2
{
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
ORBVocabulary; 
} //namespace ORB_SLAM2
#endif