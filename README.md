# S3FD for face detecttion

## 性能
|归一化尺寸|Batchsize|进程数| detect时间(ms) |平均单张时间(ms) |显存占用(GB) | 计算资源占用(%)
|:---|:---|:---|:---|:---|:---|:---
320x180| 1 | 1| 14 | 14 | 1.3 | 65
320x180| 2 | 1| 22 | 11 | 1.3 | 68
320x180| 3 | 1| 27 | 9 | 1.3 | 75
320x180| 4 | 1| 32 | 8 | 1.3 | 78
320x180| 5 | 1| 38 | 7.4 | 1.3 |78
 | | | | |
480x270| 1 | 1| 14 | 14 | 1.3 | 65
480x270| 2 | 1| 34 | 17 | 1.3 | 68
480x270| 3 | 1| 45 | 15 | 1.3 | 75
480x270| 4 | 1| 58 | 14.5 | 1.3 | 78
480x270| 5 | 1| 70 | 14 | 1.6 |80

## 使用
* 安装caffe及其依赖
http://gitlab.bmi/VisionAI/soft/caffe/tree/bmi-beta

* 配置环境
INCLUDEPATH+=$CAFFE_CODE_DIR/include
LIBS += -L$CAFFE_CODE_DIR/build/lib
LIBS+=-lcaffe  -lcurand -lcudart -lcublas \
        -lglog -lgflags -lboost_system

* 调用
引用.h, .cpp
```C++
#include "sfd.h"
#include <opencv2/opencv.hpp>
float confThresh=0.8;
int batchSize = 4;
cv::Size normalizedSize = cv::Size(480, 270);
detector = new SFD();
detector->init("SFD_deploy.prototxt", "SFD_weights.caffemodel", normalizedSize, batchSize, confThresh);

vector<Mat> imgBatch;
vector<vector<Rect> > facesBatch;
detector->detect(imgBatch, facesBatch, scoresBatch);  //目标检测,同时保存每个框的置信度

```
