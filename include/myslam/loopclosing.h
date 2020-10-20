#ifndef MYSLAM_LOOPCLOSING_H
#define MYSLAM_LOOPCLOSING_H

#include "myslam/common_include.h"
#include <opencv2/features2d.hpp>
#include "DBoW3/DBoW3.h"
#include "myslam/frame.h"
#include "myslam/camera.h"
#include "myslam/mappoint.h"
#include "myslam/algorithm.h"

namespace myslam {

class Map;

class LoopClosing {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<LoopClosing> Ptr;

    // Start the loop closure detection thread and keep it
    LoopClosing(DBoW3::Vocabulary* vocabulary);

    // Start the detection once
    void DetectLoop(Frame::Ptr frame);

    // Stop the loop closure detection thread
    void Stop();

    void SetMap(std::shared_ptr<Map> map) {map_ = map; }

    void SetORBExtractor(cv::Ptr<cv::ORB> orb) { orb_ = orb; }

    void SetCamera(Camera::Ptr camera) {camera_left_ = camera; }

private:
    // The main function of loop closure detection
    void Run();

    // Covert the type of descriptors to use for computing BoW
    std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

    // Compute BoW vector for current_keyframe_
    void ComputeBoW();

    // Compute the minscore of covisible key-frames and select the non-covisible frames larges than minScore
    float ComputeCovisibleMinScore(std::set<Frame::Ptr>& candidateKF);

    // Compute the score of bowvector a and b
    float ComputeScore(const DBoW3::BowVector &a, const DBoW3::BowVector &b) const; 

    // Detect whether there is a loop
    bool DetectLoop();

    // Compute the min common words of noncovisible key-frames and curr_frame_, used as threshold
    // Returns the candidateKF whose common words larger than mincommonwords
    float ComputeNoncovisibleMinCommonWords(const std::set<Frame::Ptr>& inputCandidateKF, 
                                            std::set<Frame::Ptr>& outputCandidateKF);

    // Compute the min group words of inputCandidateKF and return the outputCandidateKF
    float ComputeMinGroupScore(const std::set<Frame::Ptr>& inputCandidateKF, 
                                std::set<Frame::Ptr>& outputCandidateKF);

    // Select the consistent keyframes in different detection time
    void SelectConsistentKFs(const std::set<Frame::Ptr>& inputCandidateKF,
                             std::set<Frame::Ptr>& outputCandidateKF);

    // Check whether group1 and group2 has common member
    bool HasCommonMember(const std::set<Frame::Ptr>& group1, const std::set<Frame::Ptr>& group2);

    // Calculate the pose change between the loop frame and curr_keyframe_, return true if the loop frame meets the requirements 
    bool ComputeLoopPoseChange(const std::set<Frame::Ptr>& candidateKF);

    /* Use 3D-2D BA VO to estimate pose change from loop frame to current frame
       3D points are the mappoints of loop frame
       2D points are the features of curr_keyframe_
       Initial estimate value is the loop frame's pose
       Final estimate value is the T_{cw}
       return number of inliers
    */
    int CalcPoseChange(const Frame::Ptr& loopFrame, 
                       const std::unordered_map<MapPoint::Ptr, cv::Point2f>& match_3d_2d_pt, 
                       SE3& pose);

    // The thread of loop closure detection
    std::thread loopclosing_thread_;

    // The flag to indicate whether the thread is running
    std::atomic<bool> loopclosing_running_;

    // Lock of current thread. Used by std::unique_lock
    std::mutex data_mutex_;

    // The variable to trigger detect once
    std::condition_variable map_update_;

    // New insert key frame
    Frame::Ptr curr_keyframe_;
    // ORB extractor
    cv::Ptr<cv::ORB> orb_;
    // The keypoints of current left frame
    std::vector<cv::KeyPoint> curr_keypoints_left_;
    // The descriptors of current left frame
    cv::Mat curr_descriptors_left_;
    // ORB descriptors matcher
    cv::FlannBasedMatcher   matcher_flann_;     // flann matcher

    // ORB dictionary
    std::shared_ptr<DBoW3::Vocabulary> mpORBvocabulary_;

    // Contains the frames for each word
    std::vector<std::list<Frame::Ptr>> wordObservedFrames_;

    // Contains the groups with length from last time detection
    std::vector<std::pair<std::set<Frame::Ptr>, int>> prevGroupLenCounter_;

    // Camera model, used to get camera's parameteres
    Camera::Ptr camera_left_;

    // The keyframe has a loop with curr_keyframe_
    Frame::Ptr loopFrame_;
    // The pose from keyframe to curr_keyframe_
    SE3 loopToCurrentPose_;

    // The threshold to determine the outliers
    double chi2_th_ = 6.0f;

    // Map
    std::shared_ptr<Map> map_;

    // ID of the last frame use to detect loop
    unsigned long lastLoopID_ = 0;
};

} // namespace

#endif // LOOPCLOSING_H