#ifndef SFD_H
#define SFD_H

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

#ifdef DEBUG_TIME
#define calTime(start, end) ( (1000000*(end.tv_sec-start.tv_sec) + end.tv_usec-start.tv_usec) * 1.0 / 1000 )

#endif

class SFD {

public:

    /**
     * @brief SFD
     * @param modelsPath path that contains the predefied model file (SFD_deploy.prototxt)
     *          and trained weights file (SFD_weights.caffemodel)
     * @param confThresh confidence thresh when output face bbox.
     *          ie., only output bboxes whose confidence is larger than confThresh, default is 0.8
     * @param maxSide the longest side when feed image to the net
     */
    SFD(const std::string &modelsPath, float confThresh = 0.8, int maxSide = 480);

    /**
     * @brief SFD
     * @param modelFile .prototxt file
     * @param weightFile .caffemodel file
     * @param confThresh
     * @param maxSide
     */
    SFD(const std::string modelFile, const std::string weightFile, float confThresh = 0.8, int maxSide = 480);

    ~SFD(){}

    /**
     * @brief detect 检测人脸
     * @param img input picture on which to find faces
     * @param rects output, boundingbox list of all located faces
     */
    void detect(const cv::Mat& img, std::vector<cv::Rect >& rects);

    /**
     * @brief detect
     * @param img
     * @param rects
     * @param confidences
     */
    void detect(const cv::Mat& img, std::vector<cv::Rect >& rects, std::vector<float>& confidences);


private:

    void initNet(const std::string model_file, const std::string weights_file);

    void forwardNet(const cv::Mat& img);

    void getDetectResult(std::vector<cv::Rect >*rects, std::vector<float>* confidences);

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
