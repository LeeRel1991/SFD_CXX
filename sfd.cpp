#include "sfd.h"
#include <sys/time.h>
using namespace std;
using namespace caffe;
using namespace cv;

#define DEFAULT_HEIGHT 1080
#define DEFAULT_WIDTH 1920

SFD::SFD(const std::string &modelsPath):
    m_fConfThresh(0.8),
    img_max_side(480)
{
    string modelName = "deploy.prototxt";
    string weightName = "SFD.caffemodel";
    if( modelsPath[modelsPath.size() - 1 ] != '/' )
    {
        initNet(modelsPath + "/" + modelName, modelsPath + "/" + weightName);
    }
    else
        initNet(modelsPath + modelName, modelsPath + weightName);

}

SFD::SFD(const std::string modelFile, const std::string weightFile):
    m_fConfThresh(0.8),
    img_max_side(480)
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
    struct timeval st_tm, end_tm;
    static float total_time = 0.0;
    gettimeofday(&st_tm, NULL);

    //resize img and normalize
    Mat normalizedImg;
    preprocess(img, normalizedImg);

    gettimeofday(&end_tm, NULL);
    total_time = calTime( st_tm, end_tm);
    std::cerr << "process time: " << total_time << std::endl;

    std::vector<cv::Mat> input_channels;
    wrapInputLayer(&input_channels);

    gettimeofday(&end_tm, NULL);
    total_time = calTime( st_tm, end_tm);
    std::cerr << "wrap time: " << total_time << std::endl;

    // set data to net and do forward
    split(normalizedImg, input_channels);
    net_->Forward();

    gettimeofday(&end_tm, NULL);
    total_time = calTime( st_tm, end_tm);
    std::cerr << "forward time: " << total_time << std::endl;


    /* Copy the output layer to a std::vector */
    Blob<float>* result_blob = net_->output_blobs()[0];
    const float* result = result_blob->cpu_data();
    const int num_det = result_blob->height();

    for(int i=0; i<num_det; ++i, result +=7 )
    {
        if(result[0]==-1 || result[2]< m_fConfThresh)
            continue;

        int x1 = static_cast<int>(result[3] * img.cols);
        int y1 = static_cast<int>(result[4] * img.rows);
        int x2 = static_cast<int>(result[5] * img.cols);
        int y2 = static_cast<int>(result[6] * img.rows);
        Rect rect(x1, y1, x2-x1, y2-y1);
        rects.push_back(rect);

    }
    gettimeofday(&end_tm, NULL);
    total_time = calTime( st_tm, end_tm);
    std::cerr << "total time: " << total_time << std::endl;
}


/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void SFD::wrapInputLayer(std::vector<cv::Mat>* input_channels)
{
    Blob<float>* input_layer = net_->input_blobs()[0];

    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += input_geometry_.width * input_geometry_.height;
    }
}

void SFD::preprocess(const Mat &img, Mat& processedImg) {

    //计算图像缩放尺度
    Mat sample_resized;
    cv::resize(img, sample_resized, input_geometry_, 0, 0);
    sample_resized.convertTo(sample_resized, CV_32FC3);
    cv::subtract(sample_resized, mean_, processedImg);

}
