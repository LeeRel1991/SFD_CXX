#include <QCoreApplication>

#include "sfd.h"

using namespace caffe;
using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    SFD* detector = new SFD("/media/lirui/Personal/DeepLearning/FaceDetect/SFD/models/VGGNet/WIDER_FACE/SFD_trained");

    cv::VideoCapture capture;
    capture.open("/media/lirui/Program/Datas/Videos/2018-04-10-3member.mp4");
    if (!capture.isOpened())
    {
       std::cout << "视频读取失败！" << std::endl;
    }

    std::clock_t start, end;
    double total_time =0;

    Mat imgFrame;
    int ii=0;
    while(true)
    {

        capture >> imgFrame;
        if (imgFrame.empty())
            break;

        map<string, vector<float> > score;
        start = std::clock();

        vector<Rect>  label_objs;
        detector->detect(imgFrame, label_objs);  //目标检测,同时保存每个框的置信度

        end = std::clock();
        total_time = (double)(end -start) / CLOCKS_PER_SEC;
        std::cout << "time: " << total_time << "s" <<std::endl;


        for(vector<Rect>::iterator it=label_objs.begin();it!=label_objs.end();it++){
            rectangle(imgFrame, *it, Scalar(0,0,255),2);   //画出矩形框
//            putText(imgFrame, label, Point(rects[j].x,rects[j].y),CV_FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,0)); //标记 类别：置信度

        }

        imshow("1", imgFrame);
        waitKey(1);

        ii++;

    }

    delete detector;

    return 0;
}



