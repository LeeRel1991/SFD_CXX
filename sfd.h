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

    /** @overload
     * @param modelFile .prototxt file
     * @param weightFile .caffemodel file
     * @param confThresh
     * @param maxSide
     */
    SFD(const std::string modelFile, const std::string weightFile, float confThresh=0.8, int maxSide=480);

    /** @overload
     * @param modelsPath path that contains the predefied model file (SFD_deploy.prototxt)
     *          and trained weights file (SFD_weights.caffemodel)
     * @param confThresh confidence thresh when output face bbox.
     *          ie., only output bboxes whose confidence is larger than confThresh, default is 0.8
     * @param maxSide the longest side when feed image to the net
     */
    SFD(const std::string &modelsPath, float confThresh=0.8, int maxSide=480);

    ~SFD(){}

    /**
     * @brief detect 检测人脸
     * @param img input picture to locate faces
     * @param rects output the pixel boundingbox Rect of each faces on @em img
     */
    void detect(const cv::Mat& img, std::vector<cv::Rect>& rects);

    /** @overload
     * @brief detect
     * @param img input picture to locate faces
     * @param rects output the pixel boundingbox Rect of each faces (bbox) on @em img
     * @param confidences output the confidence or score (0~1) to evaluate each boundingbox in @em rects is face
     */
    void detect(const cv::Mat& img, std::vector<cv::Rect>& rects, std::vector<float>& confidences);

    /**
     * @brief detect detect face on several pictures once a time, also called batch detect
     * @param imgBatch input 一组待检测图片列表
     * @param rectsBatch output 一组每张图片检测到的boundingbox列表，其size与 @em imgBatch.size() 相等
     */
    void detect(const std::vector<cv::Mat>& imgBatch, std::vector<std::vector<cv::Rect> >& rectsBatch);

    /**
     * @brief detect
     * @param imgBatch
     * @param rectsBatch
     * @param confidencesBatch output 一组每张图片检测到的boundingbox置信概率列表，其size与 @em imgBatch.size() 相等
     */
    void detect(const std::vector<cv::Mat>& imgBatch, std::vector<std::vector<cv::Rect> >& rectsBatch, std::vector<std::vector<float> >& confidencesBatch);

private:

    void initNet(const std::string model_file, const std::string weights_file);

    void forwardNet(const std::vector<cv::Mat>& imgs);

    void getDetectResult(std::vector<std::vector<cv::Rect> >*rects, std::vector<std::vector<float> >* confidences);

    void wrapInputLayer(std::vector<cv::Mat>* input_channels);

    void preprocess(const cv::Mat& img, cv::Mat& processedImg);

    boost::shared_ptr<caffe::Net<float> > m_ptrNet;
    cv::Size m_inputGeometry;
    cv::Mat m_meanImg;
    int m_maxInSide;
    int m_numChannels;
    float m_fConfThresh;
};


#endif // SFD_H
