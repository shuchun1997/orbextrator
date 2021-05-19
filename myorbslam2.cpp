#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <unistd.h>
#include<System.h>


using namespace std;
using namespace cv;
using namespace ORB_SLAM2;
const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    string strAssociationFilename = string(argv[4]);
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
     {   cerr << endl << "No images found in provided path." << endl;
        return 1;
     }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth " <<endl;
        return 1;
    
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);


    cv::Mat imRGB,imD,seg;
    for (int i = 0; i < nImages; i++)
    {
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[i],CV_LOAD_IMAGE_UNCHANGED);
        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[i],CV_LOAD_IMAGE_UNCHANGED);
        seg = cv::imread(string(argv[5])+"/"+vstrImageFilenamesRGB[i],0);
        double trframe = vTimestamps[i];
        if(imRGB.empty() || imD.empty())
        {
            cerr << endl << "Failed to load image at:"
            << string(argv[3]) << "/" << vstrImageFilenamesRGB[i]<< endl;
            return 1;

        }

        //todo::start
// Dilation settings
    int dilation_size = 15;
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                           cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                           cv::Point( dilation_size, dilation_size ) );
        // Segment out the images
        cv::Mat mask = cv::Mat::ones(480,640,CV_8U);
        
        cv::Mat maskRCNN;
        maskRCNN = seg.clone();
            
        cv::Mat maskRCNNdil = maskRCNN.clone();
        cv::dilate(maskRCNN,maskRCNNdil, kernel);
        mask = mask - maskRCNNdil;
        cv::Mat imGray = imRGB.clone();
        imRGB = imRGB*0;
        imGray.copyTo(imRGB,mask);

        //translation images to SLAM system
        SLAM.TrackRGBD(imRGB,imD,mask,trframe);
    }    
    //stop all thread
    SLAM.Shutdown();
    return 0;
    
}

























void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}
