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
        cout<<"usage: \n "
          "./samples --modelFile $prototxt_model_file --weightFile $trained_weight_file --video $video_file_to_detect --confThresh $thresh --maxSide $max_side\n" << endl;
        cout<<" or: \n"
          "./samples --modelPath path to contains all the required files(model. weight) --video $video_file_to_detect --confThresh $thresh --maxSide $max_side\n" << endl;


        cout<<"eg: ./samples --modelFile ./SFD_trained/SFD_deploy.prototxt --weightFile ./SFD_trained/SFD_weights.caffemodel --video /media/lirui/Program/Datas/Videos/face.mp4"<<endl;
        cout<<"or:\n"
              "./samples --modelPath /media/lirui/Personal/DeepLearning/FaceDetect/SFD/models/VGGNet/WIDER_FACE/SFD_trained  "
                        "--video /media/lirui/Program/Datas/Videos/Face201701052.mp4 --confThresh 8 --maxSide 400"<<endl;

        return -1;
    }

    SFD* detector;
    string videoFile;
    float confThresh;
    int maxSide;

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

        detector = new SFD(modelPath, confThresh, maxSide);

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

            detector= new SFD(modelFile, weightFile, confThresh, maxSide );

        }
    }


    cv::VideoCapture capture;
    capture.open(videoFile);
    if (!capture.isOpened())
    {
       std::cout << "视频读取失败！" << std::endl;
    }


    Mat imgFrame;
    int ii=0;
    while(true)
    {

        capture >> imgFrame;
        if (imgFrame.empty())
            break;

#ifdef DEBUG_TIME
        struct timeval st_tm, end_tm;
        static float total_time = 0.0;
        gettimeofday(&st_tm, NULL);
#endif

        vector<vector<Rect> > label_objs;
        vector<vector<float> >scores;
        vector<Mat> imgBatch;
        for(int i=0; i<5;i++)
            imgBatch.push_back(imgFrame);

//        vector<Rect> objs;
//        detector->detect(imgFrame, objs);  //目标检测,同时保存每个框的置信度
        detector->detectBatch(imgBatch, label_objs);  //目标检测,同时保存每个框的置信度

#ifdef DEBUG_TIME
        gettimeofday(&end_tm, NULL);
        total_time = calTime( st_tm, end_tm);
        std::cout << "detect time: " << total_time << std::endl;
#endif

        for(int img_id = 0; img_id < imgBatch.size(); ++img_id)
        {
            vector<Rect> curr_objs = label_objs[img_id];
//            vector<float> curr_scores = scores[img_id];
            for(int i=0; i<curr_objs.size(); i++){
                rectangle(imgFrame, curr_objs[i], Scalar(0,0,255),2);   //画出矩形框
//                stringstream stream;
//                stream << curr_scores[i];
//                putText(imgBatch[img_id], stream.str(), Point(curr_objs[i].x, curr_objs[i].y), CV_FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,0)); //标记 类别：置信度

            }
            stringstream stream;
            stream << "img_" << img_id;
            imshow(stream.str(), imgBatch[img_id]);

        }

        waitKey(1);

        ii++;

    }

    delete detector;

    return 0;
}



