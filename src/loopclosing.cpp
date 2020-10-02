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
        // When the mpa_update is notified by other thread, the data_mutex_ will be locked
        map_update_.wait(lock);

        ComputeBoW();
        ComputeScore();
    }
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
    mpORBvocabulary_->transform(descriptors_left, curr_keyframe_->BowVec_);
}

// Compute the score of current keyframe and previous keyframes
void LoopClosing::ComputeScore()
{
    if (map_->GetAllKeyFrames().size() < 10)
        return;
    
    // Calculate the max score of latest 9 keyframes
    // Calculate the max score of previous keyframes
    float max_score = 0, refScore = 0;
    unsigned long max_id = 0;
    for(auto& kf: map_->GetAllKeyFrames())
    {
        if (curr_keyframe_->keyframe_id_ == kf.first)
            continue;
        if (curr_keyframe_->keyframe_id_ - kf.first < 10)
        {
            float score = mpORBvocabulary_->score(curr_keyframe_->BowVec_, kf.second->BowVec_);
            refScore = (score > refScore) ? score : refScore;
        }
        else
        {
            float score = mpORBvocabulary_->score(curr_keyframe_->BowVec_, kf.second->BowVec_);
            if (score > max_score)
            {
                max_score = score;
                max_id = kf.first;
            }
        }
    }

    // Select the keyframe with score larger than 2 times of refscore
    if (refScore < 1e-6 || max_score < 2 * refScore)
        return;

    LOG(INFO) << "Current:" << curr_keyframe_->keyframe_id_ << ' ' 
                << "Ref:" << refScore << ' '
                << "Looped:" << max_id << ' ' 
                << "Score:" << max_score / refScore;
}


} // namespace