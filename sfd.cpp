#include "sfd.h"
#include <sys/time.h>
using namespace std;
using namespace caffe;
using namespace cv;

#define DEFAULT_HEIGHT 1080
#define DEFAULT_WIDTH 1920

SFD::SFD(const std::string &modelsPath, float confThresh, int maxSide):
    m_fConfThresh(confThresh),
    img_max_side(maxSide)
{
    string modelName = "SFD_deploy.prototxt";
    string weightName = "SFD_weights.caffemodel";
    if( modelsPath[modelsPath.size() - 1 ] != '/' )
    {
        initNet(modelsPath + "/" + modelName, modelsPath + "/" + weightName);
    }
    else
        initNet(modelsPath + modelName, modelsPath + weightName);

}

SFD::SFD(const std::string modelFile, const std::string weightFile, float confThresh, int maxSide):
    m_fConfThresh(confThresh),
    img_max_side(maxSide)
{
    initNet(modelFile, weightFile);
}


void SFD::initNet(const std::string model_file, const std::string weights_file)
{
#ifndef CPU_ONLY
    Caffe::set_mode(Caffe::GPU);
#else
    Caffe::set_mode(Caffe::CPU);
#endif

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(weights_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    double im_shrink = double(img_max_side) / DEFAULT_WIDTH;
    input_geometry_ = Size(DEFAULT_WIDTH * im_shrink, DEFAULT_HEIGHT * im_shrink);

    mean_ = Mat(input_geometry_, CV_32FC3, cv::Scalar(104,117,123) );

    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, 3, input_geometry_.height, input_geometry_.width);
    net_->Reshape();

}

void SFD::detect(const cv::Mat& img, std::vector<cv::Rect >& rects)
{
#ifdef DEBUG_TIME
    struct timeval st_tm, end_tm;
    static float total_time = 0.0;
    gettimeofday(&st_tm, NULL);
#endif
    forwardNet(img);

#ifdef DEBUG_TIME
    gettimeofday(&end_tm, NULL);
    total_time = calTime( st_tm, end_tm);
    std::cerr << "forward time: " << total_time << std::endl;
#endif

    getDetectResult(&rects, NULL);

#ifdef DEBUG_TIME
    gettimeofday(&end_tm, NULL);
    total_time = calTime( st_tm, end_tm);
    std::cerr << "total time: " << total_time << std::endl;
#endif
}

void SFD::detect(const cv::Mat& img, std::vector<cv::Rect >& rects, std::vector<float>& confidences)
{
#ifdef DEBUG_TIME
    struct timeval st_tm, end_tm;
    static float total_time = 0.0;
    gettimeofday(&st_tm, NULL);
#endif

    forwardNet(img);

#ifdef DEBUG_TIME
    gettimeofday(&end_tm, NULL);
    total_time = calTime( st_tm, end_tm);
    std::cerr << "forward time: " << total_time << std::endl;
#endif

    getDetectResult(&rects, &confidences );

#ifdef DEBUG_TIME
    gettimeofday(&end_tm, NULL);
    total_time = calTime( st_tm, end_tm);
    std::cerr << "total time: " << total_time << std::endl;
#endif
}

void SFD::getDetectResult(std::vector<Rect>*rects, std::vector<float>*confidences)
{
    /* Copy the output layer to a std::vector */
    Blob<float>* result_blob = net_->output_blobs()[0];
    const float* result = result_blob->cpu_data();
    const int num_det = result_blob->height();

    for(int i=0; i<num_det; ++i, result +=7 )
    {
        if(result[0]==-1 || result[2]< m_fConfThresh)
            continue;

        int x1 = static_cast<int>(result[3] * DEFAULT_WIDTH);
        int y1 = static_cast<int>(result[4] * DEFAULT_HEIGHT);
        int x2 = static_cast<int>(result[5] * DEFAULT_WIDTH);
        int y2 = static_cast<int>(result[6] * DEFAULT_HEIGHT);
        Rect rect(x1, y1, x2-x1, y2-y1);
        rects->push_back(rect);
        if(confidences)
            confidences->push_back(result[2]);

    }
}

void SFD::forwardNet(const cv::Mat& img)
{

    //resize img and normalize
    Mat normalizedImg;
    preprocess(img, normalizedImg);

    std::vector<cv::Mat> input_channels;
    wrapInputLayer(&input_channels);

    // set data to net and do forward
    split(normalizedImg, input_channels);
    net_->Forward();

}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void SFD::wrapInputLayer(std::vector<cv::Mat>* input_channels)
{
    caffe::Blob<float>* input_layer = net_->input_blobs()[0];
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += input_geometry_.width * input_geometry_.height;
    }
}

void SFD::preprocess(const cv::Mat &img, cv::Mat& processedImg) {

    //计算图像缩放尺度
    cv::Mat sample_resized;
    cv::resize(img, sample_resized, input_geometry_, 0, 0);
    sample_resized.convertTo(sample_resized, CV_32FC3);
    cv::subtract(sample_resized, mean_, processedImg);

}
