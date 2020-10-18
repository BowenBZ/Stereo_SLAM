#include "myslam/loopclosing.h"
#include<opencv2/core/core.hpp>
#include "myslam/feature.h"
#include "myslam/map.h"

namespace myslam {

// Start the loop closure detection thread and keep it
LoopClosing::LoopClosing(DBoW3::Vocabulary* vocabulary) {
    mpORBvocabulary_ = std::shared_ptr<DBoW3::Vocabulary>(vocabulary);
    wordObservedFrames_.resize(mpORBvocabulary_->size());
    loopclosing_running_.store(true);
    loopclosing_thread_ = std::thread(std::bind(&LoopClosing::Run, this));
}

// Start the detection once
void LoopClosing::DetectLoop(std::shared_ptr<Frame> frame) {
    std::unique_lock<std::mutex> lock(data_mutex_);
    curr_keyframe_ = frame;
    map_update_.notify_one();
}

// Stop the loop closure detection thread
void LoopClosing::Stop() {
    loopclosing_running_.store(false);
    loopclosing_thread_.join();
}

void LoopClosing::Run()
{
    while (loopclosing_running_.load())
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        // Stuck current thread until the map_update_ is notified by other thread 
        // This automatically unlock the data_mutex_
        // When the mpa_update is notified by other thread, the data_mutex_ will be locked
        map_update_.wait(lock);

        if(DetectLoop())
        {
            lasLoopID_ = curr_keyframe_->keyframe_id_;
        }
    }
}

// Detect whether there is a loop
bool LoopClosing::DetectLoop() {
    // Compute BowVec for keyframe
    ComputeBoW();

    LOG(INFO) << "Current ID: " << curr_keyframe_->keyframe_id_;

    if(curr_keyframe_->keyframe_id_ - lasLoopID_ < 10) {
        LOG(INFO) << "Too close to last time loop";
        return false;
    }

    // Compute the minScore between curr_keyframe_ and covisible frames
    // and select the non-covisible frames larges than minScore
    std::set<Frame::Ptr> candidateKF0;
    float minScore = ComputeCovisibleMinScore(candidateKF0);
    if(candidateKF0.empty()) {
        LOG(INFO) << "No non-covisibleframe larger than minscore";
        return false;
    }

    // Select keyframes according to minCommonWords
    std::set<Frame::Ptr> candidateKF1;
    float minCommonWords = ComputeNoncovisibleMinCommonWords(candidateKF0, candidateKF1);
    if(candidateKF1.empty()) {
        LOG(INFO) << "No keyframe larges than minCommonWords";
        return false;
    }

    // Select the keyframe that largers than min group score from candidateKF1
    std::set<Frame::Ptr> candidateKF2;
    float minGroupScore = ComputeMinGroupScore(candidateKF1, candidateKF2);
    if (candidateKF2.empty()) {
        LOG(INFO) << "No frame larges than minGroupScore";
        lastTimeGroups_.clear();
        return false;
    }

    // Select the keyframes that consistent in different detect time
    std::set<Frame::Ptr> candidateKF3;
    SelectConsistentKFs(candidateKF2, candidateKF3);
    if (candidateKF3.empty()) {
        LOG(INFO) << "No consistent group";
        return false;
    }

    LOG(INFO) << "Minscore: " << minScore << ' ' 
              << "MinCommonWords: " << minCommonWords << ' '
              << "minGroupScore: " << minGroupScore << ' '
              << "candidates: " << candidateKF3.size();
    return true;
}

void LoopClosing::ComputeBoW()
{
    std::vector<cv::KeyPoint> keypoints_left;
    for(auto& kp : curr_keyframe_->features_left_) {
        keypoints_left.push_back(kp->position_);
    }

    cv::Mat descriptors_left;
    orb_->compute(curr_keyframe_->left_img_, keypoints_left, descriptors_left);
    mpORBvocabulary_->transform(descriptors_left, curr_keyframe_->BowVec_);

    // Update the wordObservedFrames_
    for(auto& word : curr_keyframe_->BowVec_) {
        wordObservedFrames_[word.first].push_back(curr_keyframe_);
    }
}

float LoopClosing::ComputeCovisibleMinScore(std::set<Frame::Ptr>& candidateKF) {

    // Calcualte minScore in covisible frames
    float minScore = 1.0f;
    // std::string connect = "";
    // std::string scores = "";
    for(auto& kf : curr_keyframe_->GetOrderedConnectedKeyFramesVector()) {
        float score = ComputeScore(curr_keyframe_->BowVec_, kf->BowVec_);
        minScore = (score < minScore) ? score : minScore;
        // connect += " " + std::to_string(kf->keyframe_id_) + ": " + std::to_string(curr_keyframe_->GetConnectedKeyFramesCounter()[kf]);
        // scores += " " + std::to_string(score);
    }
    // LOG(INFO) << "Connected ID: weight " << connect;
    // LOG(INFO) << scores;

    // Select the non-covisible keyframe larger than minscore
    for(auto& kf : map_->GetAllKeyFrames()) {
        // Select keyframe that not belongs to covisible frames
        if (kf.first != curr_keyframe_->keyframe_id_ && 
            !curr_keyframe_->GetConnectedKeyFramesCounter().count(kf.second)) {

            float score = ComputeScore(curr_keyframe_->BowVec_, kf.second->BowVec_);
            if (score > minScore) {
                // Record the score for later use
                kf.second->BoWScore_ = score;
                candidateKF.insert(kf.second);
            }
        }
    }
    return minScore;
}

float LoopClosing::ComputeScore(const DBoW3::BowVector &a, const DBoW3::BowVector &b) const 
{
    return mpORBvocabulary_->score(a, b);
}

float LoopClosing::ComputeNoncovisibleMinCommonWords(std::set<Frame::Ptr>& inputCandidateKF, 
                                                     std::set<Frame::Ptr>& outputCandidateKF) {

    // Calculate common words for candidatePassScore
    for(auto& word : curr_keyframe_->BowVec_) {
        // Get the keyframes observed this word
        for(auto& kf : wordObservedFrames_[word.first]) {
            // Check whether this keyframe in the candidatePassScore
            if(inputCandidateKF.count(kf)) {
                if(kf->commonWordsKeyframeID == curr_keyframe_->keyframe_id_) {
                    kf->commonWordsCount++;
                }
                else {
                    kf->commonWordsKeyframeID = curr_keyframe_->keyframe_id_;
                    kf->commonWordsCount = 0;
                }
            }
        }
    }

    // Calculate max common words
    float maxCommonWords = 0;
    for(auto& kf : inputCandidateKF) {
        if(kf->commonWordsKeyframeID == curr_keyframe_->keyframe_id_ &&
           kf->commonWordsCount > maxCommonWords) {
            maxCommonWords = kf->commonWordsCount;
        }
    }

    // Set the minCommonWords
    float minCommonWords = 0.8f * maxCommonWords;

    // Select the keyframe larges than minCommonWords
    for(auto& kf : inputCandidateKF) {
        if(kf->commonWordsKeyframeID == curr_keyframe_->keyframe_id_ &&
           kf->commonWordsCount >= minCommonWords) {
            outputCandidateKF.insert(kf);
        }
    }
    return minCommonWords;
}

float LoopClosing::ComputeMinGroupScore(std::set<Frame::Ptr>& inputCandidateKF, 
                                        std::set<Frame::Ptr>& outputCandidateKF) {

    // Calculate each candidate's group
    std::list<std::pair<Frame::Ptr, float>> groupScoreCounter;
    float maxGroupScore = 0;
    for(auto& kf : inputCandidateKF) {
        float groupScore = kf->BoWScore_;
        float maxScore = kf->BoWScore_;
        Frame::Ptr frameWithLargestScore = kf;

        for(auto& co_kf : kf->GetOrderedConnectedKeyFramesVector()) {
            if(inputCandidateKF.count(co_kf)) {
                groupScore += co_kf->BoWScore_;
                if(co_kf->BoWScore_ > maxScore) {
                    maxScore = co_kf->BoWScore_;
                    frameWithLargestScore = co_kf;
                }
            }
        }
        groupScoreCounter.push_back(std::make_pair(frameWithLargestScore, groupScore));
        maxGroupScore = (groupScore > maxGroupScore) ? groupScore : maxGroupScore;
    }

    // Select keyframe according to minGroupScore
    float minGroupScore = 0.75f * maxGroupScore;
    Frame::Ptr loopFrame;
    std::set<Frame::Ptr> candidateKFfromGroup;
    for(auto& kf : groupScoreCounter) {
        if(kf.second >= minGroupScore) {
            outputCandidateKF.insert(kf.first);
        }
    }
    return minGroupScore;
}


// Select the consistent keyframes in different detection time
void LoopClosing::SelectConsistentKFs(std::set<Frame::Ptr>& inputCandidateKF,
                                      std::set<Frame::Ptr>& outputCandidateKF) {

    std::vector<std::pair<std::set<Frame::Ptr>, int>> tmpThisTimeGroup;
    int minConsistencyThreshold = 3;

    for(auto& kf : inputCandidateKF) {
        std::set<Frame::Ptr> currentGroup = kf->GetConnectedKeyFramesSet();
        currentGroup.insert(kf);

        int currentConsistency = 0;

        for(auto& lastGroup : lastTimeGroups_) {
            if (HasCommonMember(currentGroup, lastGroup.first)) {
                currentConsistency = 
                    (lastGroup.second + 1 > currentConsistency) ? lastGroup.second + 1 : currentConsistency;
            }
        }
        tmpThisTimeGroup.push_back(std::make_pair(currentGroup, currentConsistency));
        
        if(currentConsistency >= minConsistencyThreshold) {
            outputCandidateKF.insert(kf);
        }
    }
    lastTimeGroups_ = tmpThisTimeGroup;
}

bool LoopClosing::HasCommonMember(const std::set<Frame::Ptr>& group1, 
                                  const std::set<Frame::Ptr>& group2) {
    for(auto& item : group1) {
        if(group2.count(item)) {
            return true;
        }
    }
    return false;
}

} // namespace