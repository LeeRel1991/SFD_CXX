#include "sfd.h"
#include <sys/time.h>
using namespace std;
using namespace caffe;
using namespace cv;


#define badBbox(result)   (result[0]==-1 || result[2]< m_fConfThresh)

cv::Scalar SFD::m_meanVector = cv::Scalar(104,117,123);

void SFD::init(const string &modelsPath, const Size imgSize,
               const int batchSize, const int gpuDevice, const float confThresh)
{

    m_batchSize = batchSize;
    m_inputGeometry = imgSize;
    m_fConfThresh = confThresh;

    string modelName = "SFD_deploy.prototxt";
    string weightName = "SFD_weights.caffemodel";
    if( modelsPath[modelsPath.size() - 1 ] != '/' )
        initNet(modelsPath + "/" + modelName, modelsPath + "/" + weightName, gpuDevice);    
    else
        initNet(modelsPath + modelName, modelsPath + weightName, gpuDevice);
}


void SFD::init(const string modelFile, const string weightFile, const Size imgSize,
               const int batchSize, const int gpuDevice, const float confThresh)
{

    m_batchSize = batchSize;
    m_inputGeometry = imgSize;
    m_fConfThresh = confThresh;
    initNet(modelFile, weightFile, gpuDevice);
}

void SFD::initNet(const string model_file, const string weights_file, const int gpuDevice)
{
#ifndef CPU_ONLY
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(gpuDevice);
#else
    Caffe::set_mode(Caffe::CPU);
#endif

    /* Load the network. */
    m_ptrNet.reset(new Net<float>(model_file, TEST));
    m_ptrNet->CopyTrainedLayersFrom(weights_file);

    CHECK_EQ(m_ptrNet->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(m_ptrNet->num_outputs(), 1) << "Network should have exactly one output.";

    m_numChannels = 3;

    Blob<float>* input_layer = m_ptrNet->input_blobs()[0];
    input_layer->Reshape(m_batchSize, m_numChannels, m_inputGeometry.height, m_inputGeometry.width);
    m_ptrNet->Reshape();
}

void SFD::detect(const Mat& img, vector<Rect >& rects)
{
    vector<Mat> imgBatch(1, img);
    vector<vector<Rect> > rectsBatch;
    detect(imgBatch, rectsBatch);
    rects.assign( rectsBatch[0].begin(), rectsBatch[0].end());
}

void SFD::detect(const Mat& img, vector<Rect>& rects, vector<float>& confidences)
{

    detect(img, rects);

    vector<vector<float> > confBatch(1);
    getConfidences(confBatch);
    confidences.assign( confBatch[0].begin(), confBatch[0].end());
}


void SFD::detect(const std::vector<cv::Mat>& imgBatch, std::vector<std::vector<cv::Rect> >& rectsBatch,
                 std::vector<std::vector<float> >& confidencesBatch)
{

    detect(imgBatch, rectsBatch);

    if(confidencesBatch.empty())
        confidencesBatch.resize(imgBatch.size());

    getConfidences(confidencesBatch);

}

void SFD::detect(const std::vector<cv::Mat>& imgBatch, std::vector<std::vector<Rect> > &rectsBatch)
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
    if(rectsBatch.size()!=imgBatch.size())
        rectsBatch.resize(imgBatch.size());

    getDetectResult(rectsBatch);

#ifdef DEBUG_TIME
    gettimeofday(&end_tm, NULL);
    total_time = calTime( st_tm, end_tm);
    std::cerr << "total time: " << total_time << std::endl;
#endif
}


void SFD::getDetectResult(vector<vector<Rect> >& rects)
{

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = m_ptrNet->output_blobs()[0];
    const float* result = output_layer->cpu_data();
    const int num_det = output_layer->height();


    for(int i=0; i<num_det; ++i, result +=7 )
    {
        int img_id = result[0];
        if(badBbox(result))
            continue;

        int x1 = static_cast<int>(result[3] * m_inputGeometry.width);
        int y1 = static_cast<int>(result[4] * m_inputGeometry.height);
        int x2 = static_cast<int>(result[5] * m_inputGeometry.width);
        int y2 = static_cast<int>(result[6] * m_inputGeometry.height);
        Rect rect(x1, y1, x2-x1, y2-y1);
        rects[img_id].push_back(rect);
    }

}

void SFD::getConfidences(std::vector<std::vector<float> > &confidences)
{

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = m_ptrNet->output_blobs()[0];
    const float* result = output_layer->cpu_data();
    const int num_det = output_layer->height();


    for(int i=0; i<num_det; ++i, result +=7 )
    {
        int img_id = result[0];
        if(badBbox(result))
            continue;
        confidences[img_id].push_back(result[2]);
    }
}

void SFD::forwardNet(const std::vector<cv::Mat>& imgs)
{
    std::vector<cv::Mat> input_data;
    wrapInputLayer(&input_data);

    for (int i = 0; i < imgs.size(); i++)
    {
        //resize img and normalize

        Mat normalizedImg = imgs[i];
        // TODO: 对输入图像的预处理在detect调用之前做
        //preprocess(imgs[i], normalizedImg);

#ifdef ENABLE_CHECK
        CHECK( normalizedImg.size() == m_inputGeometry ) << \
                "input image size must be " << m_inputGeometry.width << " x " << m_inputGeometry.height ;
        CHECK( normalizedImg.type() == CV_32FC3 ) << "input image type must be CV_32FC3";
        CHECK( normalizedImg.channels() == m_numChannels ) << "input image channel must be " << m_numChannels;
#endif
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

void SFD::preprocess(const cv::Mat&img,  cv::Mat& processedImg) {

    //计算图像缩放尺度
    cv::Mat sample_resized;
    cv::Size normalizedSize= Size(processedImg.cols, processedImg.rows);
    cv::Mat meanImg = Mat(normalizedSize, CV_32FC3, m_meanVector );
    cv::resize(img, sample_resized, normalizedSize, 0, 0);
    sample_resized.convertTo(sample_resized, CV_32FC3);
    cv::subtract(sample_resized, meanImg, processedImg);

}
