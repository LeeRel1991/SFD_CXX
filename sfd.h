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
/**
 * @brief The SFD class
 * @note 深度定制for AIStreamer
 */

class SFD {

public:

    /**
     * @note 必须显示调用init方法
     */
    SFD(){}

    ~SFD(){}

    void init(const std::string &modelsPath, const cv::Size imgSize, const int batchSize=1, const float confThresh=0.8);

    /**
     * @brief init 初始化，使用SFD()无参构造时必须显示调用该方法
     * @param modelFile .prototxt
     * @param weightFile .caffemodel
     * @param imgSize 归一化图像尺寸
     * @param batchSize 一次处理的图像个数
     * @param confThresh 阈值，影响检测框的准确度，默认直0.8
     */
    void init(const std::string modelFile, const std::string weightFile,
              const cv::Size imgSize, const int batchSize=1, const float confThresh=0.8);

    /**
     * @brief detect 检测人脸
     * @param img input picture to locate faces
     * @param rects output the pixel boundingbox Rect of each faces on @em img
     * @note 调用之前必须实现对img的预处理，参考preprocess
     * resize，转为CV_32FC3, 减去均值,
     * m_meanImg = Mat(m_inputGeometry, CV_32FC3, cv::Scalar(104,117,123) );
     *
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
    void detect(const std::vector<cv::Mat>& imgBatch, std::vector<std::vector<cv::Rect> >& rectsBatch,
                std::vector<std::vector<float> >& confidencesBatch);

    /**
     * @brief preprocess 对原始图片按照进行预处理，包括resize，resize，转为CV_32FC3, 减去均值
     * @param img input
     * @param inputSize 归一化尺寸，必须与init时设置的尺寸一致
     * @param processedImg output,size
     */
    static void preprocess(const cv::Mat& img, cv::Mat& processedImg);

    static cv::Scalar m_meanVector;

private:

    void initNet(const std::string model_file, const std::string weights_file);

    void forwardNet(const std::vector<cv::Mat>& imgs);

    void getDetectResult(std::vector<std::vector<cv::Rect> >*rects, std::vector<std::vector<float> >* confidences);

    void wrapInputLayer(std::vector<cv::Mat>* input_channels);

    boost::shared_ptr<caffe::Net<float> > m_ptrNet;
    cv::Size m_inputGeometry;

    /**
     * @brief m_batchSize 批处理时一次接收的图片数
     */
    int m_batchSize;
    int m_maxInSide;
    int m_numChannels;
    float m_fConfThresh;
};


#endif // SFD_H
