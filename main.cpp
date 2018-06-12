#include <QCoreApplication>

#include "sfd.h"
#include <time.h>

using namespace caffe;
using namespace cv;
using namespace std;


void sigleVideo()
{}

// batch detect. conduct detection on several images once a time
void multiVideo()
{

}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);   

    if (argc < 9)
    {
        cout<<"usage: \n ";
        cout<<"eg: ./samples --modelFile ./SFD_trained/SFD_deploy.prototxt --weightFile ./SFD_trained/SFD_weights.caffemodel --video /media/lirui/Program/Datas/Videos/face.mp4"<<endl;
        cout<<"or:\n"
              "./samples --modelPath ./SFD_trained  "
                        "--video Face201701052.mp4 --confThresh 8 --maxSide 400"<<endl;

        return -1;
    }

    SFD* detector;
    string videoFile;
    float confThresh;
    int maxSide;
    int batchSize = 4;

    Size normalizedSize = cv::Size(480, 270);
    detector = new SFD();

    // 解析命令行参数
    for (int i =0; i <argc; ++i)
    {
        if (std::string(argv[i]) == "--video"){
            videoFile = argv[i+1];
        }
        if (std::string(argv[i]) == "--confThresh"){
            confThresh = atoi(argv[i+1])/10.0;
        }

        if (std::string(argv[i]) == "--maxSide"){
            maxSide = atoi(argv[i+1]);
        }
    }

    if (argc < 11 )
    {
        std::string modelPath ;
        for (int i =0; i <argc; ++i)
        {

            if (std::string(argv[i]) == "--modelPath"){
                modelPath = argv[i+1];
            }
        }


        detector->init(modelPath, normalizedSize, batchSize, confThresh);

    }
    else
    {
        std::string modelFile, weightFile;

        for (int i =0; i <argc; ++i)
        {

            if (std::string(argv[i]) == "--modelFile"){
                modelFile = argv[i+1];
            }
            if (std::string(argv[i]) == "--weightFile"){
                weightFile = argv[i+1];
            }

            detector->init(modelFile, weightFile, normalizedSize, batchSize, confThresh );

        }
    }


    //检测
    cv::VideoCapture capture;
    capture.open(videoFile);
    if (!capture.isOpened())
    {
       std::cout << "视频读取失败！" << std::endl;
    }

    //cv::VideoCapture cap2;
    //cap2.open("/media/lirui/Program/Datas/Videos/face.mp4");
    //if(!cap2.isOpened())
    //    std::cout << "视频读取失败！" << std::endl;

    Mat imgFrame, imgFrame2;
    int ii=0;
    while(true)
    {

        capture >> imgFrame;
        //cap2 >> imgFrame2;
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
        //imgBatch.push_back(imgFrame);
        //imgBatch.push_back(imgFrame2);
        for(int i=0; i<batchSize; i++)
        {
            Mat processedImg = Mat(normalizedSize.height, normalizedSize.width, CV_32FC3);
            SFD::preprocess(imgFrame, processedImg);
            imgBatch.push_back(processedImg);
        }

#ifdef DEBUG_TIME
        gettimeofday(&end_tm, NULL);
        total_time = calTime( st_tm, end_tm);
        std::cout << "preprocess time: " << total_time << std::endl;
        gettimeofday(&st_tm, NULL);
#endif

        detector->detect(imgBatch, facesBatch, scoresBatch);  //目标检测,同时保存每个框的置信度

#ifdef DEBUG_TIME
        gettimeofday(&end_tm, NULL);
        total_time = calTime( st_tm, end_tm);
        std::cout << "detect time: " << total_time << std::endl;
#endif


        cv::Mat meanImg = Mat(normalizedSize, CV_32FC3, SFD::m_meanVector );
        for(int imgID = 0; imgID < imgBatch.size(); ++imgID)
        {
            Mat tmp = imgBatch[imgID];
            cv::add(tmp, meanImg, tmp);
            tmp.convertTo(tmp, CV_8UC3);
            vector<Rect> currFaces = facesBatch[imgID];
            vector<float> currScores = scoresBatch[imgID];
            for(int i=0; i<currFaces.size(); i++)
            {
                rectangle(tmp, currFaces[i], Scalar(0,0,255),2);   //画出矩形框
                stringstream stream;
                stream << currScores[i];
                putText(tmp, stream.str(), Point(currFaces[i].x, currFaces[i].y), CV_FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,0)); //标记 类别：置信度

            }
            stringstream stream;
            stream << "img_" << imgID;

            imshow(stream.str(), tmp);

        }

        waitKey(1);

        ii++;

    }

    delete detector;

    return 0;
}



