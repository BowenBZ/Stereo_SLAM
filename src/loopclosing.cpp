#include "myslam/loopclosing.h"
#include<opencv2/core/core.hpp>
#include "myslam/feature.h"
#include "myslam/map.h"
#include "myslam/g2o_types.h"
#include "myslam/config.h"



namespace myslam {

// Start the loop closure detection thread and keep it
LoopClosing::LoopClosing(DBoW3::Vocabulary* vocabulary) {
    mpORBvocabulary_ = std::shared_ptr<DBoW3::Vocabulary>(vocabulary);
    wordObservedFrames_.resize(mpORBvocabulary_->size());
    matcher_flann_ = cv::FlannBasedMatcher(new cv::flann::LshIndexParams ( 5,10,2 ));
    chi2_th_ = Config::Get<double> ("chi2_th");
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
        /* Stuck current thread until the map_update_ is notified by other thread 
         This automatically unlock the data_mutex_
         When the mpa_update is notified by other thread, the data_mutex_ will be locked
        */
        map_update_.wait(lock);

        if(DetectLoop())
        {
            lastLoopID_ = curr_keyframe_->keyframe_id_;
        }
    }
}

// Detect whether there is a loop
bool LoopClosing::DetectLoop() {
    // Compute BowVec for keyframe
    ComputeBoW();

    LOG(INFO) << "Current ID: " << curr_keyframe_->keyframe_id_;

    if(curr_keyframe_->keyframe_id_ - lastLoopID_ < 10) {
        LOG(INFO) << "Too close to last time loop";
        return false;
    }

    /* Compute the minScore between curr_keyframe_ and covisible frames
       and select the non-covisible frames larges than minScore
    */
    std::set<Frame::Ptr> candidateKF0;
    float minScore = ComputeCovisibleMinScore(candidateKF0);
    if(candidateKF0.empty()) {
        LOG(INFO) << "No non-covisibleframe larger than minscore";
        prevGroupLenCounter_.clear();
        return false;
    }

    // Select keyframes according to minCommonWords
    std::set<Frame::Ptr> candidateKF1;
    float minCommonWords = ComputeNoncovisibleMinCommonWords(candidateKF0, candidateKF1);
    if(candidateKF1.empty()) {
        LOG(INFO) << "No candidates larger than minCommonWords";
        prevGroupLenCounter_.clear();
        return false;
    }

    // Select the keyframe that largers than min group score from candidateKF1
    std::set<Frame::Ptr> candidateKF2;
    float minGroupScore = ComputeMinGroupScore(candidateKF1, candidateKF2);
    if (candidateKF2.empty()) {
        LOG(INFO) << "No candidates larger than minGroupScore";
        prevGroupLenCounter_.clear();
        return false;
    }

    // Select the keyframes that consistent in different detect time
    std::set<Frame::Ptr> candidateKF3;
    SelectConsistentKFs(candidateKF2, candidateKF3);
    // LOG(INFO) << "Consistent group size " << prevGroupLenCounter_.size();
    if (candidateKF3.empty()) {
        LOG(INFO) << "No consistent group";
        return false;
    }

    bool hasLoop = ComputeLoopPoseChange(candidateKF3);
    if (!hasLoop) {
        LOG(INFO) << "No loop frame meets pose change requirements";
        return false;
    }

    LOG(INFO) << "Minscore: " << minScore << ' ' 
              << "MinCommonWords: " << minCommonWords << ' '
              << "minGroupScore: " << minGroupScore << ' '
              << "LoopID: " << loopFrame_->keyframe_id_;
    return true;
}

void LoopClosing::ComputeBoW()
{
    curr_keypoints_left_.clear();
    for(auto& kp : curr_keyframe_->features_left_) {
        if(!kp->is_outlier_) {
            curr_keypoints_left_.push_back(kp->position_);
        }
    }
    orb_->compute(curr_keyframe_->left_img_, curr_keypoints_left_, curr_descriptors_left_);
    mpORBvocabulary_->transform(curr_descriptors_left_, curr_keyframe_->BowVec_);

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
    for(auto& kf : curr_keyframe_->GetConnectedKeyFramesCounter()) {
        float score = ComputeScore(curr_keyframe_->BowVec_, kf.first->BowVec_);
        minScore = (score < minScore) ? score : minScore;
        // connect += " " + std::to_string(kf->keyframe_id_) + ": " + std::to_string(curr_keyframe_->GetConnectedKeyFramesCounter()[kf]);
        // scores += " " + std::to_string(score);
    }
    // LOG(INFO) << "Connected ID: weight " << connect;
    // LOG(INFO) << scores;

    auto connectedKeyFramesCounter = curr_keyframe_->GetConnectedKeyFramesCounter();
    // Select the non-covisible keyframe larger than minscore
    for(auto& kf : map_->GetAllKeyFrames()) {
        // Select keyframe that not belongs to covisible frames
        if (kf.first != curr_keyframe_->keyframe_id_ && 
            !connectedKeyFramesCounter.count(kf.second)) {

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

float LoopClosing::ComputeNoncovisibleMinCommonWords(const std::set<Frame::Ptr>& inputCandidateKF, 
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

float LoopClosing::ComputeMinGroupScore(const std::set<Frame::Ptr>& inputCandidateKF, 
                                                std::set<Frame::Ptr>& outputCandidateKF) {

    // Calculate each candidate's group
    std::list<std::pair<Frame::Ptr, float>> groupScoreCounter;
    float maxGroupScore = 0;
    for(auto& kf : inputCandidateKF) {
        float groupScore = kf->BoWScore_;
        float maxScore = kf->BoWScore_;
        Frame::Ptr frameWithLargestScore = kf;

        for(auto& co_kf : kf->GetOrderedConnectedKeyFramesVector(10)) {
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

void LoopClosing::SelectConsistentKFs(const std::set<Frame::Ptr>& inputCandidateKF,
                                            std::set<Frame::Ptr>& outputCandidateKF) {

    std::vector<std::pair<std::set<Frame::Ptr>, int>> tmpGroupLenCounter;
    int minConsistencyThreshold = 3;

    for(auto& kf : inputCandidateKF) {
        std::set<Frame::Ptr> currentGroup = kf->GetConnectedKeyFramesSet();
        currentGroup.insert(kf);

        int currentConsistency = 0;

        for(auto& lastGroup : prevGroupLenCounter_) {
            if (HasCommonMember(currentGroup, lastGroup.first)) {
                currentConsistency = 
                    (lastGroup.second + 1 > currentConsistency) ? lastGroup.second + 1 : currentConsistency;
            }
        }
        tmpGroupLenCounter.push_back(std::make_pair(currentGroup, currentConsistency));
        
        if(currentConsistency >= minConsistencyThreshold) {
            outputCandidateKF.insert(kf);
        }
    }
    prevGroupLenCounter_ = tmpGroupLenCounter;
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

bool LoopClosing::ComputeLoopPoseChange(const std::set<Frame::Ptr>& candidateKF) {

    for(auto& kf : candidateKF) {
        std::vector<cv::KeyPoint> candidate_keypoints_left;
        cv::Mat candidate_descriptors_left;
        for(auto& kp : kf->features_left_) {
            if(!kp->is_outlier_ && !kp->map_point_.expired()) {
                candidate_keypoints_left.push_back(kp->position_);
            }
        }
        orb_->compute(kf->left_img_, candidate_keypoints_left, candidate_descriptors_left);

        // Match candidate_descriptors_left and curr_descriptors_left_
        std::vector<cv::DMatch> matches;
        matcher_flann_.match(candidate_descriptors_left, curr_descriptors_left_, matches);

        // Calculate the mindistance
        float min_dis = std::min_element (
                        matches.begin(), matches.end(),
                        [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
        {
            return m1.distance < m2.distance;
        } )->distance;

        // Establish the 3d-2d points pair
        // float distance_threshold = std::min<float> (min_dis * 2.0f, 30.0f);
        float distance_threshold = 30;
        std::unordered_map<MapPoint::Ptr, cv::Point2f> match_3d_2d_pts;
        for(auto& m : matches) {
            if (m.distance < distance_threshold) {
                auto mappoint = kf->features_left_[m.queryIdx]->map_point_.lock();
                if(mappoint && !mappoint->is_outlier_) {
                    match_3d_2d_pts[mappoint] = curr_keypoints_left_[m.trainIdx].pt;
                }
            }
        }

        SE3 updatedCurrentPose;
        int inliners = 0;
        if (match_3d_2d_pts.size() >= 4) {
            inliners = CalcPoseChange(kf, match_3d_2d_pts, updatedCurrentPose);
            LOG(INFO) << "Loop ID: " << kf->keyframe_id_ << " Min Distance: " << min_dis << " matched points: " << match_3d_2d_pts.size() << " inliners: " << inliners;
        }
        // if (match_3d_2d_pts.size() < 20) {
        //     continue;
        // }
    }

    return false;
}


int LoopClosing::CalcPoseChange(const Frame::Ptr& loopFrame, const std::unordered_map<MapPoint::Ptr, cv::Point2f>& match_3d_2d_pt, SE3& currentPose) {
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // camera pose vertex
    VertexPose *vertex_pose = new VertexPose();
    vertex_pose->setId(0);
    vertex_pose->setEstimate(loopFrame->Pose());
    optimizer.addVertex(vertex_pose);

    // K
    Mat33 K = camera_left_->K();

    // ID of edges
    int index = 1;
    std::unordered_map<EdgeProjectionPoseOnly *, bool> edges;

    // Use the features's connected mappoints (3D) and the features (2D)
    for (auto& pair_3d_2d : match_3d_2d_pt) {
        auto mappoint = pair_3d_2d.first;
        EdgeProjectionPoseOnly *edge =
                new EdgeProjectionPoseOnly(mappoint->GetPos(), K);
        edge->setId(index++);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(toVec2(pair_3d_2d.second));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber);
        optimizer.addEdge(edge);
        edges[edge] = false;
    }

    // estimate the Pose the determine the outliers
    unsigned int cnt_outlier = 0;

    // Optimize 4 * 10 times
    // During the first 2 * 10 times, set the outlier not optmized
    // During the last 2 * 10 times, remove the kernel
    for (int iteration = 0; iteration < 4; ++iteration) {
        vertex_pose->setEstimate(loopFrame->Pose());

        // Only optmize level 0 edges
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outlier = 0;

        // count the outliers
        for (auto& edge : edges) {
            auto e = edge.first;

            // Since the is_outlier_ is true, this feature is not optimized
            // computeError needs to be trigger manually
            if (edge.second) {
                e->computeError();
            }

            if (e->chi2() > chi2_th_) {
                // features[i]->is_outlier_ = true;
                edges[e] = true;
                e->setLevel(1);
                cnt_outlier++;
            } else {
                // features[i]->is_outlier_ = false;
                edges[e] = false;
                e->setLevel(0);
            };

            if (iteration == 2) {
                e->setRobustKernel(nullptr);
            }
        }

        
        if(cnt_outlier == match_3d_2d_pt.size())
            break;
    }

    // Set pose
    currentPose = vertex_pose->estimate();

    return match_3d_2d_pt.size() - cnt_outlier;
}



} // namespace