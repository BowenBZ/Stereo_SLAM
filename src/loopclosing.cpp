#include "myslam/loopclosing.h"
#include<opencv2/core/core.hpp>

namespace myslam {

void LoopClosing::Run()
{
    while (loopclosing_running_.load())
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        // Stuck current thread until the map_update_ is notified by other thread 
        // This automatically unlock the data_mutex_
        map_update_.wait(lock);
        LOG(INFO) << "Start detecting loop\n";
    }
}

// std::vector<cv::Mat> Converter::toDescriptorVector(const cv::Mat &Descriptors)
// {
//     std::vector<cv::Mat> vDesc;
//     vDesc.reserve(Descriptors.rows);
//     for (int j=0;j<Descriptors.rows;j++)
//         vDesc.push_back(Descriptors.row(j));

//     return vDesc;
// }

// void Frame::ComputeBoW()
// {
//     if(mBowVec.empty())
//     {
//         vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
//         mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
//     }
// }

} // namespace