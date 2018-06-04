#include "sfd.h"
#include <sys/time.h>
using namespace std;
using namespace caffe;
using namespace cv;

#define DEFAULT_HEIGHT 1080
#define DEFAULT_WIDTH 1920

SFD::SFD(const std::string &modelsPath, float confThresh, int maxSide):
    m_fConfThresh(confThresh),
    m_maxInSide(maxSide)
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

SFD::SFD(const string modelFile, const string weightFile, float confThresh, int maxSide):
    m_fConfThresh(confThresh),
    m_maxInSide(maxSide)
{
    initNet(modelFile, weightFile);
}


void SFD::initNet(const string model_file, const string weights_file)
{
#ifndef CPU_ONLY
    Caffe::set_mode(Caffe::GPU);
#else
    Caffe::set_mode(Caffe::CPU);
#endif

    /* Load the network. */
    m_ptrNet.reset(new Net<float>(model_file, TEST));
    m_ptrNet->CopyTrainedLayersFrom(weights_file);

    CHECK_EQ(m_ptrNet->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(m_ptrNet->num_outputs(), 1) << "Network should have exactly one output.";

    double im_shrink = double(m_maxInSide) / DEFAULT_WIDTH;
    m_inputGeometry = Size(DEFAULT_WIDTH * im_shrink, DEFAULT_HEIGHT * im_shrink);

    m_meanImg = Mat(m_inputGeometry, CV_32FC3, cv::Scalar(104,117,123) );
    m_numChannels = 3;
}

void SFD::detect(const Mat& img, vector<Rect >& rects)
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

    vector<vector<Rect> > rectsBatch(1, rects);
    getDetectResult(&rectsBatch, NULL);

#ifdef DEBUG_TIME
    gettimeofday(&end_tm, NULL);
    total_time = calTime( st_tm, end_tm);
    std::cerr << "total time: " << total_time << std::endl;
#endif
}

void SFD::detect(const Mat& img, vector<Rect>& rects, vector<float>& confidences)
{
#ifdef DEBUG_TIME
    struct timeval st_tm, end_tm;
    static float total_time = 0.0;
    gettimeofday(&st_tm, NULL);
#endif

    vector<Mat> imgs(1, img);
    forwardNet(imgs);

#ifdef DEBUG_TIME
    gettimeofday(&end_tm, NULL);
    total_time = calTime( st_tm, end_tm);
    std::cerr << "forward time: " << total_time << std::endl;
#endif

    vector<vector<Rect> > rectsBatch(1, rects);
    vector<vector<float> > confidencesBatch(1, confidences);
    getDetectResult(&rectsBatch, &confidencesBatch );

#ifdef DEBUG_TIME
    gettimeofday(&end_tm, NULL);
    total_time = calTime( st_tm, end_tm);
    std::cerr << "total time: " << total_time << std::endl;
#endif
}

void SFD::detect(const std::vector<cv::Mat>& imgBatch, std::vector<std::vector<cv::Rect> >& rectsBatch)
{
#ifdef DEBUG_TIME
    struct timeval st_tm, end_tm;
    static float total_time = 0.0;
    gettimeofday(&st_tm, NULL);
#endif
    forwardNet(imgBatch);

#ifdef DEBUG_TIME
    gettimeofday(&end_tm, NULL);
    total_time = calTime( st_tm, end_tm);
    std::cerr << "forward time: " << total_time << std::endl;
#endif

    // 初始化每个img对应的输出
    for(int img_id = 0; img_id < imgBatch.size(); ++img_id)
    {
        vector<Rect> faces_per_img;
        rectsBatch.push_back(faces_per_img);
    }

    getDetectResult(&rectsBatch, NULL);

#ifdef DEBUG_TIME
    gettimeofday(&end_tm, NULL);
    total_time = calTime( st_tm, end_tm);
    std::cerr << "total time: " << total_time << std::endl;
#endif
}


void SFD::detect(const std::vector<cv::Mat>& imgBatch, std::vector<std::vector<cv::Rect> >& rectsBatch, std::vector<std::vector<float> >& confidencesBatch)
{
#ifdef DEBUG_TIME
    struct timeval st_tm, end_tm;
    static float total_time = 0.0;
    gettimeofday(&st_tm, NULL);
#endif
    forwardNet(imgBatch);

#ifdef DEBUG_TIME
    gettimeofday(&end_tm, NULL);
    total_time = calTime( st_tm, end_tm);
    std::cerr << "forward time: " << total_time << std::endl;
#endif

    // 初始化每个img对应的输出
    for(int img_id = 0; img_id < imgBatch.size(); ++img_id)
    {
        vector<Rect> faces_per_img;
        rectsBatch.push_back(faces_per_img);

        vector<float> confidence_per_img;
        confidencesBatch.push_back(confidence_per_img);
    }

    getDetectResult(&rectsBatch, &confidencesBatch);

#ifdef DEBUG_TIME
    gettimeofday(&end_tm, NULL);
    total_time = calTime( st_tm, end_tm);
    std::cerr << "total time: " << total_time << std::endl;
#endif
}

void SFD::getDetectResult(vector<vector<Rect> > *rects, vector<vector<float> > *confidences)
{

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = m_ptrNet->output_blobs()[0];
    const float* result = output_layer->cpu_data();
    const int num_det = output_layer->height();


    for(int i=0; i<num_det; ++i, result +=7 )
    {
        int img_id = result[0];
        if(result[0]==-1 || result[2]< m_fConfThresh)
            continue;

        int x1 = static_cast<int>(result[3] * DEFAULT_WIDTH);
        int y1 = static_cast<int>(result[4] * DEFAULT_HEIGHT);
        int x2 = static_cast<int>(result[5] * DEFAULT_WIDTH);
        int y2 = static_cast<int>(result[6] * DEFAULT_HEIGHT);
        Rect rect(x1, y1, x2-x1, y2-y1);
        (*rects)[img_id].push_back(rect);
        if(confidences)
            (*confidences)[img_id].push_back(result[2]);

    }

}

void SFD::forwardNet(const std::vector<cv::Mat>& imgs)
{

    Blob<float>* input_layer = m_ptrNet->input_blobs()[0];
    input_layer->Reshape(imgs.size(), m_numChannels, m_inputGeometry.height, m_inputGeometry.width);
    m_ptrNet->Reshape();

    std::vector<cv::Mat> input_data;
    wrapInputLayer(&input_data);

    for (int i = 0; i < imgs.size(); i++)
    {
        //resize img and normalize
        Mat normalizedImg;
        preprocess(imgs[i], normalizedImg);
        // set data to net and do forward

        std::vector<cv::Mat>tmp_channls(3);
        tmp_channls.assign(input_data.begin() + i*m_numChannels, input_data.begin() + (i + 1)*m_numChannels);
        split(normalizedImg, tmp_channls);
    }
    m_ptrNet->Forward();

}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void SFD::wrapInputLayer(std::vector<cv::Mat>* input_channels)
{
    caffe::Blob<float>* input_layer = m_ptrNet->input_blobs()[0];
    float* input_data = input_layer->mutable_cpu_data();
    for(int j = 0; j < input_layer->num(); j++)
    {
        for (int i = 0; i < input_layer->channels(); ++i)
        {
            cv::Mat channel(m_inputGeometry.height, m_inputGeometry.width, CV_32FC1, input_data);
            input_channels->push_back(channel);
            input_data += m_inputGeometry.width * m_inputGeometry.height;
        }
    }
}

void SFD::preprocess(const cv::Mat &img, cv::Mat& processedImg) {

    //计算图像缩放尺度
    cv::Mat sample_resized;
    cv::resize(img, sample_resized, m_inputGeometry, 0, 0);
    sample_resized.convertTo(sample_resized, CV_32FC3);
    cv::subtract(sample_resized, m_meanImg, processedImg);

}
