//
// Created by gaoxiang on 19-5-2.
//

#include "myslam/backend.h"
#include "myslam/algorithm.h"
#include "myslam/feature.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/mappoint.h"

namespace myslam {

Backend::Backend() {
    backend_running_.store(true);
    backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
}

void Backend::UpdateMap(Frame::Ptr frame) {
    std::unique_lock<std::mutex> lock(data_mutex_);
    current_keyframe_ = frame;
    map_update_.notify_one();
}

void Backend::Stop() {
    backend_running_.store(false);
    map_update_.notify_one();
    backend_thread_.join();
}

void Backend::BackendLoop() {
    while (backend_running_.load()) {
        std::unique_lock<std::mutex> lock(data_mutex_);
        map_update_.wait(lock);

        /// 后端仅优化激活的Frames和Landmarks
        Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
        Map::LandmarksType active_mappoints = map_->GetActiveMapPoints();
        Optimize(active_kfs, active_mappoints);

        // Update covisible graph since some mappoint may be deleted
        // Prepard for the loopclosing
        current_keyframe_->UpdateCovisibleConnections();

        // Let loop closing detect loop because we have a new keyframe
        loopclosing_->DetectLoop(current_keyframe_);
    }
}

// BA for poses and space points
void Backend::Optimize(Map::KeyframesType &keyframes,
                       Map::LandmarksType &mappoints) {

    // LOG(INFO) << "Start backend optimizing: keyframes: " << keyframes.size() << " mappoints: " << mappoints.size();

    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // Add pose vertices
    std::map<unsigned long, VertexPose *> verticesPoseMap;
    unsigned long max_kf_id = 0;
    for (auto &keyframe : keyframes) {
        auto kf = keyframe.second;
        VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
        vertex_pose->setId(kf->keyframe_id_);
        vertex_pose->setEstimate(kf->Pose());
        optimizer.addVertex(vertex_pose);
        if (kf->keyframe_id_ > max_kf_id) {
            max_kf_id = kf->keyframe_id_;
        }

        // Record in map
        verticesPoseMap[kf->keyframe_id_] = vertex_pose;
    }

    // K 和左右外参
    Mat33 K = cam_left_->K();
    SE3 left_ext = cam_left_->pose();
    SE3 right_ext = cam_right_->pose();

    // ID of edges
    int index = 1;
    // robust kernel 阈值
    double chi2_th = 5.991;

    // Mappoint vertex
    std::map<unsigned long, VertexXYZ *> verticesMappointMap;

    // Edge and features
    std::map<EdgeProjection *, Feature::Ptr> edges_and_features;

    for (auto &mappoint : mappoints) {
        // if (mappoint.second->is_outlier_) 
        //     continue;

        // Iterates all observed features from active keyframes to build edges connecting keyframe and this mappoint
        for (auto &ob : mappoint.second->GetActiveObs()) {
            auto feat = ob.lock();
            if (feat == nullptr || feat->is_outlier_) 
                continue;

            auto frame = feat->frame_.lock();
            if (frame == nullptr)
                continue;

            // Add mappoint vertex
            unsigned long mappoint_id = mappoint.first;
            if (!verticesMappointMap.count(mappoint_id)) {
                VertexXYZ *v = new VertexXYZ;
                v->setEstimate(mappoint.second->GetPos());
                v->setId(mappoint_id + max_kf_id + 1);
                v->setMarginalized(true);
                optimizer.addVertex(v);

                // Record in map
                verticesMappointMap[mappoint_id] = v;
            }

            // Add edge
            EdgeProjection *edge = nullptr;
            if (feat->is_on_left_image_) {
                edge = new EdgeProjection(K, left_ext);
            } else {
                edge = new EdgeProjection(K, right_ext);
            }

            // Set two connecting vertex
            edge->setVertex(0, verticesPoseMap[frame->keyframe_id_]);    // pose vertex
            edge->setVertex(1, verticesMappointMap[mappoint_id]);   // mappoint vertex

            edge->setId(index++);
            edge->setMeasurement(toVec2(feat->position_.pt));
            edge->setInformation(Mat22::Identity());
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(chi2_th);
            edge->setRobustKernel(rk);
            optimizer.addEdge(edge);

            // Record edges in map
            edges_and_features[edge] = feat;
        }
    }

    // do optimization and eliminate the outliers
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // Calculate a threshold to make inliners ratio > 0.5
    int cnt_outlier = 0, cnt_inlier = 0;
    int iteration = 0;
    while (iteration < 5) {
        cnt_outlier = 0;
        cnt_inlier = 0;
        // determine if we want to adjust the outlier threshold
        for (auto &ef : edges_and_features) {
            if (ef.first->chi2() > chi2_th) {
                cnt_outlier++;
            } else {
                cnt_inlier++;
            }
        }
        double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
        if (inlier_ratio > 0.5) {
            break;
        } else {
            chi2_th *= 2;
            iteration++;
        }
    }

    // TODO: maybe set is_outlier for mappoint?
    // set is_outlier for feature since the edge between the pose and this mappoint cannot be optimized 
    // outlier features doesn't have a assigned mappoint and won't participate in the optimization
    // the mappoint should remove the observations from outlier feature should be deleted
    for (auto &ef : edges_and_features) {
        if (ef.first->chi2() > chi2_th) {
            ef.second->is_outlier_ = true;
            auto mappoint = ef.second->map_point_.lock();
            if (mappoint) {
                mappoint->RemoveKFObservation(ef.second);
            }
        } else {
            ef.second->is_outlier_ = false;
        }
    }

    // Set pose and lanrmark position
    for (auto &v : verticesPoseMap) {
        keyframes[v.first]->SetPose(v.second->estimate());
    }
    for (auto &v : verticesMappointMap) {
        mappoints[v.first]->SetPos(v.second->estimate());
    }
}

}  // namespace myslam