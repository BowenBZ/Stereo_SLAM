#pragma once

#ifndef MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

#include "myslam/camera.h"
#include "myslam/common_include.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/types.hpp>
#include "DBoW3/DBoW3.h"

namespace myslam {

// forward declare
struct MapPoint;
struct Feature;
 
/**
 * 帧
 * 每一帧分配独立id，关键帧分配关键帧ID
 */
struct Frame {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;

    unsigned long id_ = 0;           // id of this frame
    unsigned long keyframe_id_ = 0;  // id of key frame
    bool is_keyframe_ = false;       // 是否为关键帧
    double time_stamp_;              // 时间戳，暂不使用
    SE3 pose_;                       // Tcw 形式Pose
    std::mutex pose_mutex_;          // Pose数据锁
    cv::Mat left_img_, right_img_;   // stereo images

    // extracted features in left image
    std::vector<std::shared_ptr<Feature>> features_left_;
    // corresponding features in right image, set to nullptr if no corresponding
    std::vector<std::shared_ptr<Feature>> features_right_;

private:
    // Lock for connected frames
    std::mutex connectedframe_mutex_;
    // Ordered connected keyframes from large weight to small
    std::vector<Frame::Ptr> orderedConnectedKeyFrames_;
    // Connected keyframes (has same observed mappoints) (weight>15 common mappoints) and the weight
    std::unordered_map<Frame::Ptr, int> connectedKeyFramesCounter_;

public:
    // Bag of Words Vector structures.
    // 内部实际存储的是std::map<WordId, WordValue>
    // WordId 和 WordValue 表示Word在叶子中的id 和权重
    DBoW3::BowVector BowVec_;

    // The score of this frame with current detected key-frame, used by loopclosing
    float BoWScore_;
    // Has common words with the keyframe with this ID, used by loopclosing 
    unsigned long commonWordsKeyframeID;
    // How many common words does this frame has with the frame commonWordsKeyFrameID, used by loopclosing
    int commonWordsCount;

public:  // data members
    Frame() {}

    Frame(long id, double time_stamp, const SE3 &pose, const Mat &left,
          const Mat &right);

    // set and get pose, thread safe
    SE3 Pose() {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        return pose_;
    }

    void SetPose(const SE3 &pose) {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        pose_ = pose;
    }

    /// 设置关键帧并分配并键帧id
    void SetKeyFrame();

    // Update the co-visible key-frames when this frame is a key-frame 
    void UpdateConnections();

    // Get the connected keyframes as set
    std::set<Frame::Ptr> GetConnectedKeyFramesSet();

    // Get the ordered connected keyframes vector
    std::vector<Frame::Ptr> GetOrderedConnectedKeyFramesVector() {
        std::unique_lock<std::mutex> lock(connectedframe_mutex_);
        return orderedConnectedKeyFrames_;
    }

    // Get the connected keyframes counter
    std::unordered_map<Frame::Ptr, int> GetConnectedKeyFramesCounter() {
        std::unique_lock<std::mutex> lock(connectedframe_mutex_);
        return connectedKeyFramesCounter_;
    } 

    /// 工厂构建模式，分配id 
    static std::shared_ptr<Frame> CreateFrame();

private:
    // Add the connection of frame with weight to current frame
    void AddConnection(Frame::Ptr frame, const int& weight);

    // Sort the orderedConnectedFrames
    void ResortConnectedKeyframes();
};

}  // namespace myslam

#endif  // MYSLAM_FRAME_H
