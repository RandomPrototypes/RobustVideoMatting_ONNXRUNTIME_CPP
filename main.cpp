#include <onnxruntime_cxx_api.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

void testSegmentation()
{
	cv::Size size = cv::Size(1280,720);
	int deviceID = 0;
	bool use_CUDA = true;
	
	Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "segmentation");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    
    if(use_CUDA) {
    	OrtCUDAProviderOptions cuda_options;
    	sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	Ort::Session session(env, "rvm_mobilenetv3_fp32.onnx", sessionOptions);
	Ort::IoBinding io_binding(session);
	
	Ort::AllocatorWithDefaultOptions allocator;


    cv::VideoCapture cap;
	cap.open(deviceID);
	if (!cap.isOpened()) {
		printf("can not open device %d\n", deviceID);
		return ;
	}
	
	cap.set(cv::CAP_PROP_FRAME_WIDTH, size.width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, size.height);
	
	printf("create tensors\n");

	
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	
	Ort::MemoryInfo memoryInfoCuda("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
	
	std::vector<float> src_data(size.width * size.height * 3);
	std::vector<int64_t> src_dims = {1, 3, size.height, size.width};
	Ort::Value src_tensor = Ort::Value::CreateTensor<float>(memoryInfo, src_data.data(), src_data.size(), src_dims.data(), 4);
	
	float downsample_ratio = 0.25f;
	int64_t downsample_ratio_dims[] = {1};
	Ort::Value downsample_ratio_tensor = Ort::Value::CreateTensor<float>(memoryInfo, &downsample_ratio, 1, downsample_ratio_dims, 1);
	
	float rec_data = 0.0f;
	int64_t rec_dims[] = {1, 1, 1, 1};
	Ort::Value r1i = Ort::Value::CreateTensor<float>(memoryInfo, &rec_data, 1, rec_dims, 4);
	
	io_binding.BindOutput("fgr", memoryInfoCuda);
	io_binding.BindOutput("pha", memoryInfo);
	io_binding.BindOutput("r1o", memoryInfoCuda);
	io_binding.BindOutput("r2o", memoryInfoCuda);
	io_binding.BindOutput("r3o", memoryInfoCuda);
	io_binding.BindOutput("r4o", memoryInfoCuda);
    
	io_binding.BindInput("r1i", r1i);
    io_binding.BindInput("r2i", r1i);
    io_binding.BindInput("r3i", r1i);
    io_binding.BindInput("r4i", r1i);
    io_binding.BindInput("downsample_ratio", downsample_ratio_tensor);
        
	printf("start\n");
    cv::Mat frame;
    while(true) 
	{
		cap.read(frame);
		if (frame.empty()) {
            printf("error : empty frame grabbed");
            break;
        }
        
        cv::Mat blobMat;
        cv::dnn::blobFromImage(frame, blobMat);
        
        src_data.assign(blobMat.begin<float>(), blobMat.end<float>());
        for(size_t i = 0; i < src_data.size(); i++)
        	src_data[i] /= 255;
        
        io_binding.BindInput("src", src_tensor);
        session.Run(Ort::RunOptions{nullptr}, io_binding);
        
        std::vector<std::string> outputNames = io_binding.GetOutputNames();
        std::vector<Ort::Value> outputValues = io_binding.GetOutputValues();
        
        cv::Mat mask(size.height, size.width, CV_8UC1);
        for(int i = 0; i < outputNames.size(); i++) {
        	if(outputNames[i] == "pha") {
        		const cv::Mat outputImg(size.height, size.width, CV_32FC1, const_cast<float*>(outputValues[i].GetTensorData<float>()));
        		outputImg.convertTo(mask, CV_8UC1, 255.0);
        	} else if(outputNames[i] == "r1o") {
        		io_binding.BindInput("r1i", outputValues[i]);
        	} else if(outputNames[i] == "r2o") {
        		io_binding.BindInput("r2i", outputValues[i]);
        	} else if(outputNames[i] == "r3o") {
        		io_binding.BindInput("r3i", outputValues[i]);
        	} else if(outputNames[i] == "r4o") {
        		io_binding.BindInput("r4i", outputValues[i]);
        	}
        }
        cv::Mat img;
        cv::bitwise_and(frame, frame, img, mask);
        cv::imshow("img", img);
        cv::imshow("mask", mask);
        int key = cv::waitKey(10);
        if(key > 0)
        	break;
        
        
	}
}

int main(int argc, char **argv) 
{
	testSegmentation();
    return 0;
}
