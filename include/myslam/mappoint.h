#pragma once
#ifndef MYSLAM_MAPPOINT_H
#define MYSLAM_MAPPOINT_H

#include "myslam/common_include.h"

namespace myslam {

struct Frame;

struct Feature;

/**
 * 路标点类
 * 特征点在三角化之后形成路标点
 */
struct MapPoint {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<MapPoint> Ptr;

    unsigned long id_ = 0;  // ID

    // when is_outlider_ is true, that means the observations is 0. no feature could observed this mappoint
    bool is_outlier_ = false;

    MapPoint() {}

    MapPoint(long id, Vec3 position);

    // factory function
    static MapPoint::Ptr CreateNewMappoint();

    Vec3 GetPos() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return pos_;
    }

    void SetPos(const Vec3 &pos) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        pos_ = pos;
    };

    // Add a new observation, called by map when the map receive a new keyframe
    // The feature should only be feature_left
    void AddKFObservation(std::shared_ptr<Feature> feature);

    // Remove a observation by backend, since this feature is an outlier determined by backend
    void RemoveKFObservation(std::shared_ptr<Feature> feature);

    // Remove a observation from active keyframes' observation by map, used to help clean non-active mappoitns
    void RemoveActiveKFObservation(std::shared_ptr<Feature> feature);

    std::list<std::weak_ptr<Feature>> GetObs() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return observations_;
    }

    std::list<std::weak_ptr<Feature>> GetActiveObs() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_observations_;
    }

    // being observed times
    int GetObsCount() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return observations_.size();
    }

    // being observed by feature matching algo
    int GetActiveObsCount() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_observations_.size();
    }

private:
    std::mutex data_mutex_;

    // Position in world
    Vec3 pos_ = Vec3::Zero(); 

    // features_left from active keyframes observed this mappoint, used by map
    std::list<std::weak_ptr<Feature>> active_observations_; 

    // features_left from keyframes observed this mappoint
    std::list<std::weak_ptr<Feature>> observations_;    
};
}  // namespace myslam

#endif  // MYSLAM_MAPPOINT_H
