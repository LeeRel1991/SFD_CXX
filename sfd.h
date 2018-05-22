#ifndef SFD_H
#define SFD_H

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

#define calTime(start, end) ( (1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec) * 1.0 / 1000 )


class SFD {

public:


    SFD(const std::string &modelsPath);

    SFD(const std::string modelFile, const std::string weightFile);

    ~SFD(){}

    void detect(const cv::Mat& img, std::vector<cv::Rect >& rects);

    void detect(const cv::Mat& img, std::vector<cv::Rect >& rects, std::vector<float> confidences);


private:

    void initNet(const std::string model_file, const std::string weights_file);

    void wrapInputLayer(std::vector<cv::Mat>* input_channels);

    void preprocess(const cv::Mat& img, cv::Mat& processedImg);

    boost::shared_ptr<caffe::Net<float> > net_;
    cv::Size input_geometry_;
    cv::Mat mean_;
    int img_max_side;
    int num_channels_;
    float m_fConfThresh;
};
#endif // SFD_H
