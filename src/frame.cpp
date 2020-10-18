/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "myslam/frame.h"
#include "myslam/feature.h"
#include "myslam/mappoint.h"
#include "myslam/config.h"


namespace myslam {

Frame::Frame(long id, double time_stamp, const SE3 &pose, const Mat &left, const Mat &right)
        : id_(id), time_stamp_(time_stamp), pose_(pose), left_img_(left), right_img_(right) { }

Frame::Ptr Frame::CreateFrame() {
    static long factory_id = 0;
    Frame::Ptr new_frame(new Frame);
    new_frame->id_ = factory_id++;
    return new_frame;
}

void Frame::SetKeyFrame() {
    static long keyframe_factory_id = 0;
    is_keyframe_ = true;
    keyframe_id_ = keyframe_factory_id++;
}

void Frame::UpdateConnections() {

    // Calcualte the co-visible frames' count
    std::unordered_map<Frame::Ptr, int> KFCounter;
    for(auto& fe : features_left_) {
        auto mappoint = fe->map_point_.lock();
        if(mappoint) {
            for(auto& observed_fea_ptr : mappoint->observations_) {

                auto observed_fea = observed_fea_ptr.lock();
                if(observed_fea && observed_fea->is_on_left_image_) {

                    auto observed_frame = observed_fea->frame_.lock();
                    if(observed_frame && observed_frame->keyframe_id_ != this->keyframe_id_) {
                        KFCounter[observed_frame]++;
                    }
                }
            }
        }
    }
    if(KFCounter.empty())
        return;

    // Filter out the connections whose weight less than 15
    int maxCount = 0;
    Frame::Ptr maxCountKF;
    std::vector<std::pair<int, Frame::Ptr>> connectedKF;
    for(auto& kf : KFCounter) {
        if(kf.second >= 15) {
            connectedKF.push_back(std::make_pair(kf.second, kf.first));
            kf.first->AddConnection(Ptr(this), kf.second);
        }
    }
    if(connectedKF.empty()) {
        connectedKF.push_back(std::make_pair(maxCount, maxCountKF));
        maxCountKF->AddConnection(Ptr(this), maxCount);
    }

    std::unique_lock<std::mutex> lock(connectedframe_mutex_);

    // Sort the connections with weight
    std::sort(connectedKF.begin(), connectedKF.end());
    std::list<Frame::Ptr> tmpOrderedConnectedKeyFrames;
    connectedKeyFramesCounter_.clear();
    for(auto& kf : connectedKF) {
        tmpOrderedConnectedKeyFrames.push_front(kf.second);
        connectedKeyFramesCounter_[kf.second] = kf.first;
    }
    orderedConnectedKeyFrames_.clear();
    orderedConnectedKeyFrames_ = std::vector<Frame::Ptr>(tmpOrderedConnectedKeyFrames.begin(),
                                                        tmpOrderedConnectedKeyFrames.end());
}

void Frame::AddConnection(Frame::Ptr frame, const int& weight) {
    std::unique_lock<std::mutex> lock(connectedframe_mutex_);

    connectedKeyFramesCounter_[frame] = weight;
    ResortConnectedKeyframes();
}

void Frame::ResortConnectedKeyframes() {
    std::vector<std::pair<int, Frame::Ptr>> connectedKF;
    for(auto& kf : connectedKeyFramesCounter_) {
        connectedKF.push_back(std::make_pair(kf.second, kf.first));
    }

    std::sort(connectedKF.begin(), connectedKF.end());
    std::list<Frame::Ptr> tmpOrderedConnectedKeyFrames;
    connectedKeyFramesCounter_.clear();
    for(auto& kf : connectedKF) {
        tmpOrderedConnectedKeyFrames.push_front(kf.second);
        connectedKeyFramesCounter_[kf.second] = kf.first;
    }
}

std::set<Frame::Ptr> Frame::GetConnectedKeyFramesSet() {
    std::unique_lock<std::mutex> lock(connectedframe_mutex_);

    std::set<Frame::Ptr> tmp;
    for(auto& kf : connectedKeyFramesCounter_) {
        tmp.insert(kf.first);
    }
    return tmp;
}



}   // namespace
