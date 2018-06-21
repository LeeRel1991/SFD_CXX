#include "sfd.h"
#include <sys/time.h>
using namespace std;
using namespace caffe;
using namespace cv;


#define badBbox(result)   (result[0]==-1 || result[2]< m_fConfThresh)
#define checkImgProp(normalImg) \
{ \
    CHECK( normalImg.size() == m_inputGeometry ) << \
        "input image size must be " << m_inputGeometry.width << " x " << m_inputGeometry.height ; \
    CHECK( normalImg.type() == CV_32FC3 ) << "input image type must be CV_32FC3"; \
    CHECK( normalImg.channels() == m_numChannels ) << "input image channel must be " << m_numChannels; \
}

cv::Scalar SFD::m_meanVector = cv::Scalar(104,117,123);

SFD::SFD():
    m_numChannels(3)
{

}

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
    // 方式1 转为GpuMat
    vector<cuda::GpuMat> imgGpuBatch;
    for(auto &img : imgBatch)
    {
        cuda::GpuMat imgGpu;
        imgGpu.upload(img);
        imgGpuBatch.push_back(imgGpu);
    }

    detect(imgGpuBatch, rectsBatch);


    // 方式2 cpu->cpu see git commit@4fac826
}

void SFD::detect(const std::vector<cv::cuda::GpuMat>& imgBatch, std::vector<std::vector<cv::Rect> > &rectsBatch)
{
    CHECK(imgBatch.size() == m_batchSize) << "input img num must be = " << m_batchSize << endl;

#ifdef DEBUG_TIME
    struct timeval st_tm, end_tm;
    static float total_time = 0.0;
    gettimeofday(&st_tm, NULL);
#endif

    // core 核心程序
    forwardNet(imgBatch);

#ifdef DEBUG_TIME
    gettimeofday(&end_tm, NULL);
    total_time = calTime( st_tm, end_tm);
    std::cerr << "forward time: " << total_time << std::endl;
#endif

    // 获取检测bbox
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

void SFD::forwardNet(const std::vector<cuda::GpuMat> &imgs)
{
    // 初始化net inputdata, 获取inputdata的地址
    caffe::Blob<float>* input_layer = m_ptrNet->input_blobs()[0];
    float* input_data_gpu = input_layer->mutable_gpu_data();

    size_t data_len = m_inputGeometry.width * m_inputGeometry.height ;
    size_t dpitch = m_inputGeometry.width * sizeof(float);

    for(auto iter = imgs.cbegin(); iter!=imgs.cend(); iter++)
    {

#ifdef ENABLE_CHECK
        checkImgProp((*iter));
#endif
        //set data to net and do forward
        std::vector<cuda::GpuMat>tmp_channls(m_numChannels);
        cuda::split(*iter, tmp_channls);

        // 把每个通道图像从GpuMat拷贝到caffe网络的inputgpudata指向的内存
        for(auto channel: tmp_channls)
        {

            /**
             * @brief cudaMemcpy2D 2D数组内存拷贝
             * @note GpuMat内存不连续，有补全，但目标地址内存数据是连续的，
             * 因此用cudaMemcpy2D 而不是cudaMemcpy
             * @note 从GpuMat拷贝到caffe的inputgpudata 480x270耗时约0.006ms
             */
            cudaMemcpy2D(input_data_gpu, dpitch,
                         channel.data, channel.step,
                         dpitch, m_inputGeometry.height , cudaMemcpyDeviceToDevice);

            input_data_gpu += data_len;
        }
    }

    m_ptrNet->Forward();
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer.
 * 几乎耗时可忽略，约0.01ms
*/
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
