#include <QCoreApplication>

#include "sfd.h"
#include <time.h>

using namespace caffe;
using namespace cv;
using namespace std;


void sigleVideo()
{}

// batch detect. conduct detection on several images once a time
void drawDetectResults(Mat& img, vector<Rect>& bboxes, vector<float>& scores)
{

    auto itBbox = bboxes.cbegin();
    auto itScore = scores.cbegin();
    for ( ; itBbox!=bboxes.cend(); itBbox++, itScore++  )
    {
        rectangle(img, *itBbox, Scalar(0,0,255),2);   //画出矩形框
        stringstream stream;
        stream << *itScore;
        putText(img, stream.str(), Point(itBbox->x, itBbox->y), CV_FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,0)); //标记 类别：置信度
    }
}

// batch detect. conduct detection on several images once a time
void drawDetectResults(vector<Mat>& imgBatch, vector<vector<Rect> >& rectsBatch,
                       vector<vector<float> >& scoresBatch)
{

    for(int id = 0; id < imgBatch.size(); ++id)
    {
        drawDetectResults(imgBatch[id], rectsBatch[id], scoresBatch[id]);
        stringstream stream;
        stream << "img_" << id;
        imshow(stream.str(), imgBatch[id]);
    }
}


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);   

    if (argc < 7)
    {
        cout<<"usage: \n "
          "./samples data_path path_or_file video_file max_side, conf_thresh gpu_id\n" << endl;
        cout<<" or: \n"
          "./samples ./data 0 Face201701052.mp4 400 8  0\n" << endl;

        return -1;
    }

    SFD* detector;

    string modelPath = argv[1];
    string videoFile = argv[3];
    int maxSide = atoi(argv[4]);
    float confThresh = atoi(argv[5])/10.0;
    int gpuID = atoi(argv[6]);

    if(atoi(argv[2]))
        detector = new SFD(modelPath, gpuID, confThresh, maxSide);
    else
        detector = new SFD(modelPath + "/SFD_deploy.prototxt", modelPath+"/SFD_weights.caffemodel",
                           gpuID, confThresh, maxSide);


    cv::VideoCapture capture;
    capture.open(videoFile);
    if (!capture.isOpened())
    {
       std::cout << "视频读取失败！" << std::endl;
    }

//    cv::VideoCapture cap2;
//    cap2.open("/media/lirui/Program/Datas/Videos/face.mp4");
//    if(!cap2.isOpened())
//        std::cout << "视频读取失败！" << std::endl;

    Mat imgFrame, imgFrame2;
    int ii=0;
    while(true)
    {

        capture >> imgFrame;
        // cap2 >> imgFrame2;
        if (imgFrame.empty())
            break;

        //if(imgFrame2.empty())
        //    break;

#ifdef DEBUG_TIME
        struct timeval st_tm, end_tm;
        static float total_time = 0.0;
        gettimeofday(&st_tm, NULL);
#endif

        vector<vector<Rect> > facesBatch;
        vector<vector<float> > scoresBatch;
        vector<Mat> imgBatch;
//        imgBatch.push_back(imgFrame);
//        imgBatch.push_back(imgFrame2);
        for(int i=0; i<2; i++)
            imgBatch.push_back(imgFrame);

        detector->detect(imgBatch, facesBatch, scoresBatch);  //目标检测,同时保存每个框的置信度


#ifdef DEBUG_TIME
        gettimeofday(&end_tm, NULL);
        total_time = calTime( st_tm, end_tm);
        std::cout << "detect time: " << total_time << std::endl;
#endif

        drawDetectResults(imgBatch, facesBatch, scoresBatch);
//        drawDetectResults(imgFrame, tmpRects, tmpScores);
//        imshow("im", imgFrame);

        waitKey(1);

        ii++;

    }

    delete detector;

    return 0;
}



