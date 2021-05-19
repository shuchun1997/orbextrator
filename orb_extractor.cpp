
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/calib3d/calib3d.hpp>
#include<string>
#include<nmmintrin.h>
#include<chrono>
#include<algorithm>


std::string first_file_mask = "/home/shuchun/SlamDemo/orbextrator/img1_mask.png";  
std::string second_file_mask = "/home/shuchun/SlamDemo/orbextrator/img2_mask.png";
std::string first_file = "/home/shuchun/SlamDemo/orbextrator/1341846313.553992yuantu.png";
std::string second_file = "/home/shuchun/SlamDemo/orbextrator/1341846313.592026yuantu.png";
typedef std::vector<uint32_t> DescType;
void ComputeORB(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, std::vector<DescType> &descriptors);

void BfMatch(const std::vector<DescType> &desc1, const std::vector<DescType> &desc2, std::vector<cv::DMatch> &matches);

void  pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1,
                           std::vector<cv::KeyPoint> keypoints_2,
                           std::vector<cv::DMatch> matches,
                           cv::Mat &R, cv::Mat &t);

void DrawTextInfo(cv::Mat &im,cv::Mat &imText);
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);  

void triangulation(const std::vector<cv::KeyPoint> &keypoint_1,
                  const std::vector<cv::KeyPoint> &keypoint_2,
                   const std::vector<cv::DMatch> &matches,
                   const cv::Mat &R, const cv::Mat &t,
                   std::vector<cv::Point3d> &points);

int main(int argc,char **argv)
{
    cv::Mat img1_mask = cv::imread(first_file_mask,0);//0 is gray,1 is color;
    cv::Mat img2_mask = cv::imread(second_file_mask,0);
    cv::Mat img1 = cv::imread(first_file,1);
    cv::Mat img2 = cv::imread(second_file,1);
    cv::Mat img1_yuantu = img1.clone();
    cv::Mat img2_yuantu = img2.clone();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

//todo::start

  int dilation_size = 15;
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        cv::Point( dilation_size, dilation_size ) );
    cv::Mat mask = cv::Mat::ones(480,640,CV_8U);
    cv::Mat maskRCNN;
    maskRCNN = img1_mask.clone();
    cv::Mat maskRCNNdil = maskRCNN.clone(); 
    cv::dilate(maskRCNN,maskRCNNdil, kernel);  //膨胀处理
  
    mask = mask - maskRCNNdil;//mask被处理过
    cv::Mat imGray;
    cv::cvtColor(img1, imGray, CV_BGR2GRAY);
    
    img1 = img1*0;
    img1_yuantu.copyTo(img1,mask);



    cv::imshow("img1",img1);
    cv::imshow("maskRCNNdil",maskRCNNdil);
    cv::waitKey(0);
//todo::end

    assert(img1.data != nullptr);
    assert(img2.data != nullptr);
    assert(img1_mask.data != nullptr);
    assert(img2_mask.data!= nullptr);
   // std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    //compute first_image keypoints descriptor
    std::vector<cv::KeyPoint> keypoints1;
    cv::FAST(img1,keypoints1,40); //阈值越大，提取点越少
    std::vector<DescType> descriptor1;
    ComputeORB(img1,keypoints1,descriptor1);
//todo::erode
    cv::Mat Mask_dil = mask.clone();
    cv::erode(mask,Mask_dil,kernel);
    std::vector<cv::KeyPoint> _keypoints1;
    cv::Mat _descriptor1;
    for (size_t i = 0; i < keypoints1.size(); i++)
    {
      int val = (int) Mask_dil.at<uchar>(keypoints1[i].pt.y,keypoints1[i].pt.x);
      if (val == 1)
      {
        _keypoints1.push_back(keypoints1[i]);
       // _descriptor1.push_back(descriptor1.row(i));
      }
    }
    keypoints1 = _keypoints1;
    cv::imshow("img1_yuantu ",img1_yuantu);
    cv::waitKey(0);
    cv::Mat outimg1;
//这里画特征点的时候，我们是在原图上画的，其实提取的时候有讲究，画的时候不用考虑
    //cv::drawKeypoints(img1_yuantu,keypoints1,outimg1,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);
    const float r = 5;
    const float n = keypoints1.size();  
    for (int i = 0; i < n; i++)
    {
      cv::Point2f pt1,pt2;
      pt1.x = keypoints1[i].pt.x-r;
      pt1.y = keypoints1[i].pt.y-r;
      pt2.x = keypoints1[i].pt.x+r;
      pt2.y = keypoints1[i].pt.y+r;
    
      cv::rectangle(img1_yuantu,pt1,pt2,cv::Scalar(0,255,0));
      cv::circle(img1_yuantu,keypoints1[i].pt,2,cv::Scalar(0,255,0),-1);
    }
    cv::imshow("ORB features ",img1_yuantu);
    //cv::imshow("ORB features ",outimg1);
    cv::waitKey(0);
    


//compute second_image keypoints descriptor

    cv::Mat mask2 = cv::Mat::ones(480,640,CV_8U);
    cv::Mat maskRCNN2;
    maskRCNN2 = img2_mask.clone();
    cv::Mat maskRCNNdil2 = maskRCNN2.clone(); 
    cv::dilate(maskRCNN2,maskRCNNdil2, kernel);  //膨胀处理
  
    mask2 = mask2 - maskRCNNdil2;//mask被处理过
    cv::Mat imGray2;
    cv::cvtColor(img2, imGray2, CV_BGR2GRAY);
    
    img2 = img2*0;
    imGray2.copyTo(img2,mask2);
    cv::imshow("img2",img2);
    cv::imshow("maskRCNNdil2",maskRCNNdil2);
    cv::waitKey(0);

    assert(img1.data != nullptr);
    assert(img2.data != nullptr);
    assert(img1_mask.data != nullptr);
    assert(img2_mask.data!= nullptr); 

    std::vector<cv::KeyPoint> keypoints2;
    cv::FAST(img2,keypoints2,40);
    std::vector<DescType> descriptor2;
    ComputeORB(img2,keypoints2,descriptor2);

    cv::Mat Mask_dil2 = mask2.clone();
    cv::erode(mask2,Mask_dil2,kernel);
    std::vector<cv::KeyPoint> _keypoints2;
    cv::Mat _descriptor2;
    for (size_t i = 0; i < keypoints2.size(); i++)
    {
      int val = (int) Mask_dil2.at<uchar>(keypoints2[i].pt.y,keypoints2[i].pt.x);
      if (val == 1)
      {
        _keypoints2.push_back(keypoints2[i]);
       // _descriptor2.push_back(descriptor2.row(i));
      }
    }
    keypoints2 = _keypoints2;
    cv::imshow("img2_yuantu ",img2_yuantu);
    cv::waitKey(0);
    cv::Mat outimg2;
//这里画特征点的时候，我们是在原图上画的，其实提取的时候有讲究，画的时候不用考虑
    cv::drawKeypoints(img2_yuantu,keypoints2,outimg2,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("ORB features2 ",outimg2);
    cv::waitKey(0);
    std::cout << "mask+边缘处理之后的特征点个数："<<"\n";
    std::cout << "keypoints1:"<<keypoints1.size()<<"\n";
    std::cout << "keypoints2:"<<keypoints2.size()<<std::endl;
    //std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    //std::chrono::duration<double> time_used = 
    //std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    //std::cout <<"extractor ORB cost = "<< time_used.count() <<"seconds"<< std::endl;
    std::vector<cv::DMatch> matches;
    BfMatch(descriptor1,descriptor2,matches);

  

    cv::Mat R, t;
    pose_estimation_2d2d(keypoints1, keypoints2, matches, R, t);
    std::cout << "matches:" << matches.size() << std::endl;
    cv::Mat imageshow;
   // cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, imageshow);//报错未解决
    //cv::imshow("match",image_show);
   // cv::waitKey(0);

//验证E = t^R
//把向量转换成矩阵的形式
    cv::Mat t_x = (cv:: Mat_<double>(3, 3) << 0, -t.at<double>(2,0), t.at<double>(1,0),
                  t.at<double>(2,0), 0, -t.at<double>(0,0),
                  -t.at<double>(1,0), t.at<double>(1,0), 0 );
    std::cout << "t_x = "<< t_x <<std::endl;
//计算本质矩阵E
    std::cout << "E = " << t_x * R <<std::endl;
    cv::Mat E = t_x * R;
//验证对极约束
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);  //内参矩阵
    for (cv::DMatch m: matches){
  
      cv::Point2d pt1 = pixel2cam(keypoints1[m.queryIdx].pt,K);  //像素坐标转换成相机坐标
      cv::Mat y1 = (cv::Mat_<double>(3,1) << pt1.x,pt1.y,1);
      cv::Point2d pt2 = pixel2cam(keypoints1[m.trainIdx].pt,K); //像素坐标转换成相机坐标
      cv::Mat y2 = (cv::Mat_<double>(3,1) << pt2.x, pt2.y,1);
      cv::Mat d = y2.t() * t_x * R *y1;
      std::cout <<"epipolar constraint" << d <<std::endl;
     
    }

    //三角化计算深度点
      std::vector<cv::Point3d> points_3d;
      int num =0, _num=0;
      triangulation(keypoints1,keypoints2,matches,R,t,points_3d);
      for (int i = 0; i < points_3d.size(); i++)
      {
        //std::cout<<"depth"<< points_3d[i].z<<std::endl;
        if(points_3d[i].z >0){
                num++;
        }
        else
        _num ++;
      }
      std::cout << "深度值为正的个数为："<<num << "\n";
      std::cout << "深度值为负的个数为："<<_num << std::endl;
      

    return 0;

}

// ORB pattern
int ORB_pattern[256 * 4] = {
  8, -3, 9, 5/*mean (0), correlation (0)*/,
  4, 2, 7, -12/*mean (1.12461e-05), correlation (0.0437584)*/,
  -11, 9, -8, 2/*mean (3.37382e-05), correlation (0.0617409)*/,
  7, -12, 12, -13/*mean (5.62303e-05), correlation (0.0636977)*/,
  2, -13, 2, 12/*mean (0.000134953), correlation (0.085099)*/,
  1, -7, 1, 6/*mean (0.000528565), correlation (0.0857175)*/,
  -2, -10, -2, -4/*mean (0.0188821), correlation (0.0985774)*/,
  -13, -13, -11, -8/*mean (0.0363135), correlation (0.0899616)*/,
  -13, -3, -12, -9/*mean (0.121806), correlation (0.099849)*/,
  10, 4, 11, 9/*mean (0.122065), correlation (0.093285)*/,
  -13, -8, -8, -9/*mean (0.162787), correlation (0.0942748)*/,
  -11, 7, -9, 12/*mean (0.21561), correlation (0.0974438)*/,
  7, 7, 12, 6/*mean (0.160583), correlation (0.130064)*/,
  -4, -5, -3, 0/*mean (0.228171), correlation (0.132998)*/,
  -13, 2, -12, -3/*mean (0.00997526), correlation (0.145926)*/,
  -9, 0, -7, 5/*mean (0.198234), correlation (0.143636)*/,
  12, -6, 12, -1/*mean (0.0676226), correlation (0.16689)*/,
  -3, 6, -2, 12/*mean (0.166847), correlation (0.171682)*/,
  -6, -13, -4, -8/*mean (0.101215), correlation (0.179716)*/,
  11, -13, 12, -8/*mean (0.200641), correlation (0.192279)*/,
  4, 7, 5, 1/*mean (0.205106), correlation (0.186848)*/,
  5, -3, 10, -3/*mean (0.234908), correlation (0.192319)*/,
  3, -7, 6, 12/*mean (0.0709964), correlation (0.210872)*/,
  -8, -7, -6, -2/*mean (0.0939834), correlation (0.212589)*/,
  -2, 11, -1, -10/*mean (0.127778), correlation (0.20866)*/,
  -13, 12, -8, 10/*mean (0.14783), correlation (0.206356)*/,
  -7, 3, -5, -3/*mean (0.182141), correlation (0.198942)*/,
  -4, 2, -3, 7/*mean (0.188237), correlation (0.21384)*/,
  -10, -12, -6, 11/*mean (0.14865), correlation (0.23571)*/,
  5, -12, 6, -7/*mean (0.222312), correlation (0.23324)*/,
  5, -6, 7, -1/*mean (0.229082), correlation (0.23389)*/,
  1, 0, 4, -5/*mean (0.241577), correlation (0.215286)*/,
  9, 11, 11, -13/*mean (0.00338507), correlation (0.251373)*/,
  4, 7, 4, 12/*mean (0.131005), correlation (0.257622)*/,
  2, -1, 4, 4/*mean (0.152755), correlation (0.255205)*/,
  -4, -12, -2, 7/*mean (0.182771), correlation (0.244867)*/,
  -8, -5, -7, -10/*mean (0.186898), correlation (0.23901)*/,
  4, 11, 9, 12/*mean (0.226226), correlation (0.258255)*/,
  0, -8, 1, -13/*mean (0.0897886), correlation (0.274827)*/,
  -13, -2, -8, 2/*mean (0.148774), correlation (0.28065)*/,
  -3, -2, -2, 3/*mean (0.153048), correlation (0.283063)*/,
  -6, 9, -4, -9/*mean (0.169523), correlation (0.278248)*/,
  8, 12, 10, 7/*mean (0.225337), correlation (0.282851)*/,
  0, 9, 1, 3/*mean (0.226687), correlation (0.278734)*/,
  7, -5, 11, -10/*mean (0.00693882), correlation (0.305161)*/,
  -13, -6, -11, 0/*mean (0.0227283), correlation (0.300181)*/,
  10, 7, 12, 1/*mean (0.125517), correlation (0.31089)*/,
  -6, -3, -6, 12/*mean (0.131748), correlation (0.312779)*/,
  10, -9, 12, -4/*mean (0.144827), correlation (0.292797)*/,
  -13, 8, -8, -12/*mean (0.149202), correlation (0.308918)*/,
  -13, 0, -8, -4/*mean (0.160909), correlation (0.310013)*/,
  3, 3, 7, 8/*mean (0.177755), correlation (0.309394)*/,
  5, 7, 10, -7/*mean (0.212337), correlation (0.310315)*/,
  -1, 7, 1, -12/*mean (0.214429), correlation (0.311933)*/,
  3, -10, 5, 6/*mean (0.235807), correlation (0.313104)*/,
  2, -4, 3, -10/*mean (0.00494827), correlation (0.344948)*/,
  -13, 0, -13, 5/*mean (0.0549145), correlation (0.344675)*/,
  -13, -7, -12, 12/*mean (0.103385), correlation (0.342715)*/,
  -13, 3, -11, 8/*mean (0.134222), correlation (0.322922)*/,
  -7, 12, -4, 7/*mean (0.153284), correlation (0.337061)*/,
  6, -10, 12, 8/*mean (0.154881), correlation (0.329257)*/,
  -9, -1, -7, -6/*mean (0.200967), correlation (0.33312)*/,
  -2, -5, 0, 12/*mean (0.201518), correlation (0.340635)*/,
  -12, 5, -7, 5/*mean (0.207805), correlation (0.335631)*/,
  3, -10, 8, -13/*mean (0.224438), correlation (0.34504)*/,
  -7, -7, -4, 5/*mean (0.239361), correlation (0.338053)*/,
  -3, -2, -1, -7/*mean (0.240744), correlation (0.344322)*/,
  2, 9, 5, -11/*mean (0.242949), correlation (0.34145)*/,
  -11, -13, -5, -13/*mean (0.244028), correlation (0.336861)*/,
  -1, 6, 0, -1/*mean (0.247571), correlation (0.343684)*/,
  5, -3, 5, 2/*mean (0.000697256), correlation (0.357265)*/,
  -4, -13, -4, 12/*mean (0.00213675), correlation (0.373827)*/,
  -9, -6, -9, 6/*mean (0.0126856), correlation (0.373938)*/,
  -12, -10, -8, -4/*mean (0.0152497), correlation (0.364237)*/,
  10, 2, 12, -3/*mean (0.0299933), correlation (0.345292)*/,
  7, 12, 12, 12/*mean (0.0307242), correlation (0.366299)*/,
  -7, -13, -6, 5/*mean (0.0534975), correlation (0.368357)*/,
  -4, 9, -3, 4/*mean (0.099865), correlation (0.372276)*/,
  7, -1, 12, 2/*mean (0.117083), correlation (0.364529)*/,
  -7, 6, -5, 1/*mean (0.126125), correlation (0.369606)*/,
  -13, 11, -12, 5/*mean (0.130364), correlation (0.358502)*/,
  -3, 7, -2, -6/*mean (0.131691), correlation (0.375531)*/,
  7, -8, 12, -7/*mean (0.160166), correlation (0.379508)*/,
  -13, -7, -11, -12/*mean (0.167848), correlation (0.353343)*/,
  1, -3, 12, 12/*mean (0.183378), correlation (0.371916)*/,
  2, -6, 3, 0/*mean (0.228711), correlation (0.371761)*/,
  -4, 3, -2, -13/*mean (0.247211), correlation (0.364063)*/,
  -1, -13, 1, 9/*mean (0.249325), correlation (0.378139)*/,
  7, 1, 8, -6/*mean (0.000652272), correlation (0.411682)*/,
  1, -1, 3, 12/*mean (0.00248538), correlation (0.392988)*/,
  9, 1, 12, 6/*mean (0.0206815), correlation (0.386106)*/,
  -1, -9, -1, 3/*mean (0.0364485), correlation (0.410752)*/,
  -13, -13, -10, 5/*mean (0.0376068), correlation (0.398374)*/,
  7, 7, 10, 12/*mean (0.0424202), correlation (0.405663)*/,
  12, -5, 12, 9/*mean (0.0942645), correlation (0.410422)*/,
  6, 3, 7, 11/*mean (0.1074), correlation (0.413224)*/,
  5, -13, 6, 10/*mean (0.109256), correlation (0.408646)*/,
  2, -12, 2, 3/*mean (0.131691), correlation (0.416076)*/,
  3, 8, 4, -6/*mean (0.165081), correlation (0.417569)*/,
  2, 6, 12, -13/*mean (0.171874), correlation (0.408471)*/,
  9, -12, 10, 3/*mean (0.175146), correlation (0.41296)*/,
  -8, 4, -7, 9/*mean (0.183682), correlation (0.402956)*/,
  -11, 12, -4, -6/*mean (0.184672), correlation (0.416125)*/,
  1, 12, 2, -8/*mean (0.191487), correlation (0.386696)*/,
  6, -9, 7, -4/*mean (0.192668), correlation (0.394771)*/,
  2, 3, 3, -2/*mean (0.200157), correlation (0.408303)*/,
  6, 3, 11, 0/*mean (0.204588), correlation (0.411762)*/,
  3, -3, 8, -8/*mean (0.205904), correlation (0.416294)*/,
  7, 8, 9, 3/*mean (0.213237), correlation (0.409306)*/,
  -11, -5, -6, -4/*mean (0.243444), correlation (0.395069)*/,
  -10, 11, -5, 10/*mean (0.247672), correlation (0.413392)*/,
  -5, -8, -3, 12/*mean (0.24774), correlation (0.411416)*/,
  -10, 5, -9, 0/*mean (0.00213675), correlation (0.454003)*/,
  8, -1, 12, -6/*mean (0.0293635), correlation (0.455368)*/,
  4, -6, 6, -11/*mean (0.0404971), correlation (0.457393)*/,
  -10, 12, -8, 7/*mean (0.0481107), correlation (0.448364)*/,
  4, -2, 6, 7/*mean (0.050641), correlation (0.455019)*/,
  -2, 0, -2, 12/*mean (0.0525978), correlation (0.44338)*/,
  -5, -8, -5, 2/*mean (0.0629667), correlation (0.457096)*/,
  7, -6, 10, 12/*mean (0.0653846), correlation (0.445623)*/,
  -9, -13, -8, -8/*mean (0.0858749), correlation (0.449789)*/,
  -5, -13, -5, -2/*mean (0.122402), correlation (0.450201)*/,
  8, -8, 9, -13/*mean (0.125416), correlation (0.453224)*/,
  -9, -11, -9, 0/*mean (0.130128), correlation (0.458724)*/,
  1, -8, 1, -2/*mean (0.132467), correlation (0.440133)*/,
  7, -4, 9, 1/*mean (0.132692), correlation (0.454)*/,
  -2, 1, -1, -4/*mean (0.135695), correlation (0.455739)*/,
  11, -6, 12, -11/*mean (0.142904), correlation (0.446114)*/,
  -12, -9, -6, 4/*mean (0.146165), correlation (0.451473)*/,
  3, 7, 7, 12/*mean (0.147627), correlation (0.456643)*/,
  5, 5, 10, 8/*mean (0.152901), correlation (0.455036)*/,
  0, -4, 2, 8/*mean (0.167083), correlation (0.459315)*/,
  -9, 12, -5, -13/*mean (0.173234), correlation (0.454706)*/,
  0, 7, 2, 12/*mean (0.18312), correlation (0.433855)*/,
  -1, 2, 1, 7/*mean (0.185504), correlation (0.443838)*/,
  5, 11, 7, -9/*mean (0.185706), correlation (0.451123)*/,
  3, 5, 6, -8/*mean (0.188968), correlation (0.455808)*/,
  -13, -4, -8, 9/*mean (0.191667), correlation (0.459128)*/,
  -5, 9, -3, -3/*mean (0.193196), correlation (0.458364)*/,
  -4, -7, -3, -12/*mean (0.196536), correlation (0.455782)*/,
  6, 5, 8, 0/*mean (0.1972), correlation (0.450481)*/,
  -7, 6, -6, 12/*mean (0.199438), correlation (0.458156)*/,
  -13, 6, -5, -2/*mean (0.211224), correlation (0.449548)*/,
  1, -10, 3, 10/*mean (0.211718), correlation (0.440606)*/,
  4, 1, 8, -4/*mean (0.213034), correlation (0.443177)*/,
  -2, -2, 2, -13/*mean (0.234334), correlation (0.455304)*/,
  2, -12, 12, 12/*mean (0.235684), correlation (0.443436)*/,
  -2, -13, 0, -6/*mean (0.237674), correlation (0.452525)*/,
  4, 1, 9, 3/*mean (0.23962), correlation (0.444824)*/,
  -6, -10, -3, -5/*mean (0.248459), correlation (0.439621)*/,
  -3, -13, -1, 1/*mean (0.249505), correlation (0.456666)*/,
  7, 5, 12, -11/*mean (0.00119208), correlation (0.495466)*/,
  4, -2, 5, -7/*mean (0.00372245), correlation (0.484214)*/,
  -13, 9, -9, -5/*mean (0.00741116), correlation (0.499854)*/,
  7, 1, 8, 6/*mean (0.0208952), correlation (0.499773)*/,
  7, -8, 7, 6/*mean (0.0220085), correlation (0.501609)*/,
  -7, -4, -7, 1/*mean (0.0233806), correlation (0.496568)*/,
  -8, 11, -7, -8/*mean (0.0236505), correlation (0.489719)*/,
  -13, 6, -12, -8/*mean (0.0268781), correlation (0.503487)*/,
  2, 4, 3, 9/*mean (0.0323324), correlation (0.501938)*/,
  10, -5, 12, 3/*mean (0.0399235), correlation (0.494029)*/,
  -6, -5, -6, 7/*mean (0.0420153), correlation (0.486579)*/,
  8, -3, 9, -8/*mean (0.0548021), correlation (0.484237)*/,
  2, -12, 2, 8/*mean (0.0616622), correlation (0.496642)*/,
  -11, -2, -10, 3/*mean (0.0627755), correlation (0.498563)*/,
  -12, -13, -7, -9/*mean (0.0829622), correlation (0.495491)*/,
  -11, 0, -10, -5/*mean (0.0843342), correlation (0.487146)*/,
  5, -3, 11, 8/*mean (0.0929937), correlation (0.502315)*/,
  -2, -13, -1, 12/*mean (0.113327), correlation (0.48941)*/,
  -1, -8, 0, 9/*mean (0.132119), correlation (0.467268)*/,
  -13, -11, -12, -5/*mean (0.136269), correlation (0.498771)*/,
  -10, -2, -10, 11/*mean (0.142173), correlation (0.498714)*/,
  -3, 9, -2, -13/*mean (0.144141), correlation (0.491973)*/,
  2, -3, 3, 2/*mean (0.14892), correlation (0.500782)*/,
  -9, -13, -4, 0/*mean (0.150371), correlation (0.498211)*/,
  -4, 6, -3, -10/*mean (0.152159), correlation (0.495547)*/,
  -4, 12, -2, -7/*mean (0.156152), correlation (0.496925)*/,
  -6, -11, -4, 9/*mean (0.15749), correlation (0.499222)*/,
  6, -3, 6, 11/*mean (0.159211), correlation (0.503821)*/,
  -13, 11, -5, 5/*mean (0.162427), correlation (0.501907)*/,
  11, 11, 12, 6/*mean (0.16652), correlation (0.497632)*/,
  7, -5, 12, -2/*mean (0.169141), correlation (0.484474)*/,
  -1, 12, 0, 7/*mean (0.169456), correlation (0.495339)*/,
  -4, -8, -3, -2/*mean (0.171457), correlation (0.487251)*/,
  -7, 1, -6, 7/*mean (0.175), correlation (0.500024)*/,
  -13, -12, -8, -13/*mean (0.175866), correlation (0.497523)*/,
  -7, -2, -6, -8/*mean (0.178273), correlation (0.501854)*/,
  -8, 5, -6, -9/*mean (0.181107), correlation (0.494888)*/,
  -5, -1, -4, 5/*mean (0.190227), correlation (0.482557)*/,
  -13, 7, -8, 10/*mean (0.196739), correlation (0.496503)*/,
  1, 5, 5, -13/*mean (0.19973), correlation (0.499759)*/,
  1, 0, 10, -13/*mean (0.204465), correlation (0.49873)*/,
  9, 12, 10, -1/*mean (0.209334), correlation (0.49063)*/,
  5, -8, 10, -9/*mean (0.211134), correlation (0.503011)*/,
  -1, 11, 1, -13/*mean (0.212), correlation (0.499414)*/,
  -9, -3, -6, 2/*mean (0.212168), correlation (0.480739)*/,
  -1, -10, 1, 12/*mean (0.212731), correlation (0.502523)*/,
  -13, 1, -8, -10/*mean (0.21327), correlation (0.489786)*/,
  8, -11, 10, -6/*mean (0.214159), correlation (0.488246)*/,
  2, -13, 3, -6/*mean (0.216993), correlation (0.50287)*/,
  7, -13, 12, -9/*mean (0.223639), correlation (0.470502)*/,
  -10, -10, -5, -7/*mean (0.224089), correlation (0.500852)*/,
  -10, -8, -8, -13/*mean (0.228666), correlation (0.502629)*/,
  4, -6, 8, 5/*mean (0.22906), correlation (0.498305)*/,
  3, 12, 8, -13/*mean (0.233378), correlation (0.503825)*/,
  -4, 2, -3, -3/*mean (0.234323), correlation (0.476692)*/,
  5, -13, 10, -12/*mean (0.236392), correlation (0.475462)*/,
  4, -13, 5, -1/*mean (0.236842), correlation (0.504132)*/,
  -9, 9, -4, 3/*mean (0.236977), correlation (0.497739)*/,
  0, 3, 3, -9/*mean (0.24314), correlation (0.499398)*/,
  -12, 1, -6, 1/*mean (0.243297), correlation (0.489447)*/,
  3, 2, 4, -8/*mean (0.00155196), correlation (0.553496)*/,
  -10, -10, -10, 9/*mean (0.00239541), correlation (0.54297)*/,
  8, -13, 12, 12/*mean (0.0034413), correlation (0.544361)*/,
  -8, -12, -6, -5/*mean (0.003565), correlation (0.551225)*/,
  2, 2, 3, 7/*mean (0.00835583), correlation (0.55285)*/,
  10, 6, 11, -8/*mean (0.00885065), correlation (0.540913)*/,
  6, 8, 8, -12/*mean (0.0101552), correlation (0.551085)*/,
  -7, 10, -6, 5/*mean (0.0102227), correlation (0.533635)*/,
  -3, -9, -3, 9/*mean (0.0110211), correlation (0.543121)*/,
  -1, -13, -1, 5/*mean (0.0113473), correlation (0.550173)*/,
  -3, -7, -3, 4/*mean (0.0140913), correlation (0.554774)*/,
  -8, -2, -8, 3/*mean (0.017049), correlation (0.55461)*/,
  4, 2, 12, 12/*mean (0.01778), correlation (0.546921)*/,
  2, -5, 3, 11/*mean (0.0224022), correlation (0.549667)*/,
  6, -9, 11, -13/*mean (0.029161), correlation (0.546295)*/,
  3, -1, 7, 12/*mean (0.0303081), correlation (0.548599)*/,
  11, -1, 12, 4/*mean (0.0355151), correlation (0.523943)*/,
  -3, 0, -3, 6/*mean (0.0417904), correlation (0.543395)*/,
  4, -11, 4, 12/*mean (0.0487292), correlation (0.542818)*/,
  2, -4, 2, 1/*mean (0.0575124), correlation (0.554888)*/,
  -10, -6, -8, 1/*mean (0.0594242), correlation (0.544026)*/,
  -13, 7, -11, 1/*mean (0.0597391), correlation (0.550524)*/,
  -13, 12, -11, -13/*mean (0.0608974), correlation (0.55383)*/,
  6, 0, 11, -13/*mean (0.065126), correlation (0.552006)*/,
  0, -1, 1, 4/*mean (0.074224), correlation (0.546372)*/,
  -13, 3, -9, -2/*mean (0.0808592), correlation (0.554875)*/,
  -9, 8, -6, -3/*mean (0.0883378), correlation (0.551178)*/,
  -13, -6, -8, -2/*mean (0.0901035), correlation (0.548446)*/,
  5, -9, 8, 10/*mean (0.0949843), correlation (0.554694)*/,
  2, 7, 3, -9/*mean (0.0994152), correlation (0.550979)*/,
  -1, -6, -1, -1/*mean (0.10045), correlation (0.552714)*/,
  9, 5, 11, -2/*mean (0.100686), correlation (0.552594)*/,
  11, -3, 12, -8/*mean (0.101091), correlation (0.532394)*/,
  3, 0, 3, 5/*mean (0.101147), correlation (0.525576)*/,
  -1, 4, 0, 10/*mean (0.105263), correlation (0.531498)*/,
  3, -6, 4, 5/*mean (0.110785), correlation (0.540491)*/,
  -13, 0, -10, 5/*mean (0.112798), correlation (0.536582)*/,
  5, 8, 12, 11/*mean (0.114181), correlation (0.555793)*/,
  8, 9, 9, -6/*mean (0.117431), correlation (0.553763)*/,
  7, -4, 8, -12/*mean (0.118522), correlation (0.553452)*/,
  -10, 4, -10, 9/*mean (0.12094), correlation (0.554785)*/,
  7, 3, 12, 4/*mean (0.122582), correlation (0.555825)*/,
  9, -7, 10, -2/*mean (0.124978), correlation (0.549846)*/,
  7, 0, 12, -2/*mean (0.127002), correlation (0.537452)*/,
  -1, -6, 0, -11/*mean (0.127148), correlation (0.547401)*/
};

void ComputeORB(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, std::vector<DescType> &descriptors) 
{
  const int half_patch_size = 8;
  const int half_boundary = 16;
  int bad_points = 0;
  for (auto &kp: keypoints) {
    if (kp.pt.x < half_boundary || kp.pt.y < half_boundary ||
        kp.pt.x >= img.cols - half_boundary || kp.pt.y >= img.rows - half_boundary) {
      // outside
      bad_points++;
      descriptors.push_back({});
      continue;
    }

    float m01 = 0, m10 = 0;
    for (int dx = -half_patch_size; dx < half_patch_size; ++dx) {
      for (int dy = -half_patch_size; dy < half_patch_size; ++dy) {
        uchar pixel = img.at<uchar>(kp.pt.y + dy, kp.pt.x + dx);
        m10 += dx * pixel;
        m01 += dy * pixel;
      }
    }

    // angle should be arc tan(m01/m10);
    float m_sqrt = sqrt(m01 * m01 + m10 * m10) + 1e-18; // avoid divide by zero
    float sin_theta = m01 / m_sqrt;
    float cos_theta = m10 / m_sqrt;

    // compute the angle of this point
    DescType desc(8, 0);
    for (int i = 0; i < 8; i++) {
      uint32_t d = 0;
      for (int k = 0; k < 32; k++) {
        int idx_pq = i * 32 + k;
        cv::Point2f p(ORB_pattern[idx_pq * 4], ORB_pattern[idx_pq * 4 + 1]);
        cv::Point2f q(ORB_pattern[idx_pq * 4 + 2], ORB_pattern[idx_pq * 4 + 3]);

        // rotate with theta
        cv::Point2f pp = cv::Point2f(cos_theta * p.x - sin_theta * p.y, sin_theta * p.x + cos_theta * p.y)
                         + kp.pt;
        cv::Point2f qq = cv::Point2f(cos_theta * q.x - sin_theta * q.y, sin_theta * q.x + cos_theta * q.y)
                         + kp.pt;
        if (img.at<uchar>(pp.y, pp.x) < img.at<uchar>(qq.y, qq.x)) {
          d |= 1 << k;
        }
      }
      desc[i] = d;
    }
    descriptors.push_back(desc);
  }

  std::cout << "bad/total: " << bad_points << "/" << keypoints.size() << std::endl;
}

void  pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1,
                           std::vector<cv::KeyPoint> keypoints_2,
                           std::vector<cv::DMatch> matches,
                           cv::Mat &R, cv::Mat &t){
cv::Mat K = (cv::Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 21.0, 249.7, 0, 0, 1);
//把匹配点转换为vector<Point2f>的形式
std::vector<cv::Point2f>  points1;
std::vector<cv::Point2f>  points2;

for (int i = 0; i < (int) matches.size(); i++)
{
  points1.push_back(keypoints_1[matches[i].queryIdx].pt);
  points2.push_back(keypoints_2[matches[i].trainIdx].pt);
}

// compute fundamental matrix
cv::Mat fundamental_matrix ;
fundamental_matrix = cv::findFundamentalMat(points1,points2,CV_FM_8POINT);
std::cout<< "fundamental_matrix is "<<std::endl 
         << fundamental_matrix << std::endl;

//compute Essential matrix 
cv::Point2f principal_point(325.1, 249.7);
double focal_length = 521;
cv::Mat essential_matrix = cv::findEssentialMat(points1,points2,CV_FM_8POINT);
std::cout<<"essential matrix is " <<std::endl
         <<essential_matrix<<std::endl;

//compute Homogrphy matrix
cv::Mat homography_matrix;
homography_matrix = cv::findHomography(points1,points2,cv::RANSAC,3);
std::cout << "homography matrix is " << std::endl
          << homography_matrix << std::endl;

// recovery R ,t from Essential matrix
cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
std::cout<< "R is "<< std::endl
         << R << std::endl
         <<"t is "<<std::endl
         << t << std::endl;

         }


void BfMatch(const std::vector<DescType> &desc1, const std::vector<DescType> &desc2, std::vector<cv::DMatch> &matches)
{
    const int d_max = 40; //设定一个阈值
    for (size_t i1 = 0; i1 < desc1.size(); ++i1)
    {
        if(desc1[i1].empty()) continue;
        cv::DMatch m{i1,0,256};
        for (size_t i2 = 0; i2 < desc2.size(); ++i2)
        {
            if (desc2[i2].empty()) continue;
            int distance = 0;
            for (int k = 0; k < 8; k++)  //8位
            {
                distance += _mm_popcnt_u32(desc1[i1][k] ^ desc2[i2][k]);
            }
            if(distance < d_max && distance < m.distance){
                m.distance = distance;
                m.trainIdx = i2;
            }    
        }       
        if (m.distance < d_max)
        {
            matches.push_back(m);
        }
    }
    
}


void DrawTextInfo(cv::Mat &im,cv::Mat &imText){
std::stringstream s;
int baseline=0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

    imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);



}

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K)
 {
  return cv::Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}



void triangulation(const std::vector<cv::KeyPoint> &keypoint_1,
                  const std::vector<cv::KeyPoint> &keypoint_2,
                   const std::vector<cv::DMatch> &matches,
                   const cv::Mat &R, const cv::Mat &t,
                   std::vector<cv::Point3d> &points){



                  cv::Mat T1 =(cv::Mat_<float>(3,4) <<  1,0,0,0,
                                                        0,1,0,0,
                                                        0,0,1,0,
                                                        0,0,0,1);
                  cv::Mat T2 =(cv::Mat_<float>(3,4) <<  
                          R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),t.at<double>(0,0),
                          R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),t.at<double>(1,0),
                          R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2),t.at<double>(2,0));
                  cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);  //内参矩阵
                  std::vector<cv::Point2f> pts_1, pts_2;
                  for (cv::DMatch m:matches)
                  { //像素坐标转换至相机坐标系
                    pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt,K));
                    pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt,K));
                  }
                  
                  cv::Mat pts_4d;
                  cv::triangulatePoints(T1,T2,pts_1,pts_2,pts_4d);
                  for (int i = 0; i < pts_4d.cols; i++)
                  {
                    cv::Mat x = pts_4d.col(i);   
                    x /= x.at<float>(3,0);
                    cv::Point3d p (x.at<float>(0,0),
                                  x.at<float>(1.0),
                                  x.at<float>(2,0));
                    points.push_back(p);


                  }
                  
                   }





/***
//暴力匹配代码，自己复现的，正确性待验证，
void BfMatch(const std::vector<DescType> &desc1, const std::vector<DescType> &desc2, std::vector<cv::DMatch> &matches)
{
    const int d_max = 100; //设定一个阈值
    std::vector<int> dis;
    for (size_t i1 = 0; i1 < desc1.size(); ++i1)
    {   dis.clear();
        if(desc1[i1].empty()) continue;
        cv::DMatch m{i1,0,256};
        for (size_t i2 = 0; i2 < desc2.size(); ++i2)
        {
            if (desc2[i2].empty()) continue;
            int distance = 0;
            for (int k = 0; k < 8; k++)  //8位
            {
                distance += _mm_popcnt_u32(desc1[i1][k] ^ desc2[i2][k]);
            }
            
            dis.push_back(distance);

                
        } 
        
        double minPosition = min_element(dis.begin(),dis.end()) - dis.begin();
        double minValue = dis[minPosition];
       // m.distance = minValue;
        m.trainIdx = minPosition; 
        
        matches.push_back(m);
        
    }
    
}
***/

