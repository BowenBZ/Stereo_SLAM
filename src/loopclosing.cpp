#include "myslam/loopclosing.h"
#include<opencv2/core/core.hpp>
#include "myslam/feature.h"

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
        ComputeBoW();
    }
}

std::vector<cv::Mat> LoopClosing::toDescriptorVector(const cv::Mat &Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j=0;j<Descriptors.rows;j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

// Compute BoW vector for current_keyframe_
void LoopClosing::ComputeBoW()
{
    std::vector<cv::KeyPoint> keypoints_left;
    for(auto &kp: curr_keyframe_->features_left_)
    {
        keypoints_left.push_back(kp->position_);
    }
    cv::Mat descriptors_left;
    orb_->compute(curr_keyframe_->left_img_, keypoints_left, descriptors_left);
    vector<cv::Mat> vCurrentDesc = toDescriptorVector(descriptors_left);
    mpORBvocabulary_->transform(vCurrentDesc, curr_keyframe_->mBowVec_, curr_keyframe_->mFeatVec_, 4);
}


} // namespace