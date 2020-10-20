#pragma once
#ifndef MAP_H
#define MAP_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"
#include "myslam/config.h"

namespace myslam {

/**
 * @brief 地图
 * 和地图的交互：前端调用InsertKeyframe和InsertMapPoint插入新帧和地图点，后端维护地图的结构，判定outlier/剔除等等
 */
class Map {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Map> Ptr;
    typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;
    typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;

    Map() {num_active_keyframes_ = Config::Get<int>("num_active_keyframes");}

    // Insert a new keyframe, called by frontend
    void InsertKeyFrame(Frame::Ptr frame);
    // Insert a new mappoint, called by frontend
    void InsertMapPoint(MapPoint::Ptr map_point);

    /// 获取所有地图点
    LandmarksType GetAllMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return landmarks_;
    }
    /// 获取所有关键帧
    KeyframesType GetAllKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return keyframes_;
    }

    /// 获取激活地图点
    LandmarksType GetActiveMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_landmarks_;
    }

    /// 获取激活关键帧
    KeyframesType GetActiveKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_keyframes_;
    }


   private:

    /**
     * Set the features in new keyframe as new observation of the map points
    */
    void SetMappointObservationsForKeyframe();

    // 将旧的关键帧置为不活跃状态
    void RemoveOldKeyframe();

    // Remove the active mappoints whose active_observations_ is 0
    void CleanMappoints();
    
    std::mutex data_mutex_;
    LandmarksType landmarks_;         // all landmarks
    LandmarksType active_landmarks_;  // active landmarks, maximum is 7
    KeyframesType keyframes_;         // all key-frames
    KeyframesType active_keyframes_;  // active key-frames

    Frame::Ptr current_frame_ = nullptr;

    // settings
    unsigned int num_active_keyframes_ = 7;  // 激活的关键帧数量
};
}  // namespace myslam

#endif  // MAP_H
