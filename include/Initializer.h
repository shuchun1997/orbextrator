#ifndef INITIALIZER_H
#define INITIALIZER_H

#include<opencv2/opencv.hpp>
#include<Frame.h>


namespace ORB_SLAM2
{
class Initializer
{
    public:
        Initializer(const Frame &ReferenceFrame, float sigma = 1.0, int iterations = 200);

};
}
#endif