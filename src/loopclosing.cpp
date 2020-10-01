#include "myslam/loopclosing.h"
#include<opencv2/core/core.hpp>
#include "myslam/feature.h"
#include "myslam/map.h"

namespace myslam {

void LoopClosing::Run()
{
    while (loopclosing_running_.load())
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        // Stuck current thread until the map_update_ is notified by other thread 
        // This automatically unlock the data_mutex_
        map_update_.wait(lock);
        // LOG(INFO) << "Start detecting loop\n";
        ComputeBoW();
        ComputeScore();
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

void LoopClosing::ComputeScore()
{
    if (curr_keyframe_->keyframe_id_ == 0)
        return;

    if (map_->GetAllKeyFrames().size() <= 10)
        return;

    float refScore = 0;
    for (unsigned long i = 1; i < 10; i++)
    {
        float score = mpORBvocabulary_->score(curr_keyframe_->mBowVec_, 
                                              map_->GetAllKeyFrames()[curr_keyframe_->keyframe_id_-i]->mBowVec_);
        refScore = (score > refScore) ? score : refScore;
    }
    refScore = (refScore > 1e-6) ? refScore : 1e-6;
    
    float max_score = 0;
    unsigned long max_id = 0;
    for(auto& kf: map_->GetAllKeyFrames())
    {
        if (kf.first == curr_keyframe_->keyframe_id_)
            continue;
        if (curr_keyframe_->keyframe_id_ - kf.first < 10)
            continue;
        float score = mpORBvocabulary_->score(curr_keyframe_->mBowVec_, kf.second->mBowVec_);

        if (score > max_score)
        {
            max_score = score;
            max_id = kf.first;
        }
    }
    if (max_score < 2 * refScore)
        return;

    LOG(INFO) << "Current:" << curr_keyframe_->keyframe_id_ << ' ' 
                << "Ref:" << refScore << ' '
                << "Looped:" << max_id << ' ' 
                << "Score" << max_score;
}


} // namespace