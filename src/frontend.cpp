//
// Created by gaoxiang on 19-5-2.
//

#include <opencv2/opencv.hpp>

#include "myslam/algorithm.h"
#include "myslam/backend.h"
#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/viewer.h"

namespace myslam {

Frontend::Frontend() {
    //gftt_ = cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
    orb_ = cv::ORB::create ( Config::Get<int> ( "num_features" ), 
                             Config::Get<double> ("scale_factor"), 
                             Config::Get<int> ( "level_pyramid" ) );
    num_features_init_ = Config::Get<int>("num_features_init");
    num_features_tracking_ = Config::Get<int>("num_features_tracking");
    num_features_tracking_bad_ = Config::Get<int>("num_features_tracking_bad");
    num_features_needed_for_keyframe_ = Config::Get<int>("num_features_needed_for_keyframe");
}

bool Frontend::AddFrame(myslam::Frame::Ptr frame) {
    current_frame_ = frame;
    // LOG(INFO) << "Current VO Status: " << (int)status_;

    switch (status_) {
        case FrontendStatus::INITING:
            StereoInit();
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track();
            break;
        case FrontendStatus::LOST:
            Reset();
            break;
    }

    last_frame_ = current_frame_;
    return true;
}

// Firstly set current pose as (relative_motion between last 2 frames) * pose of last frame
bool Frontend::Track() {
    if (last_frame_) {
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
    }

    FindFeaturesInCurrent();
    tracking_inliers_ = EstimateCurrentPose();

    if (tracking_inliers_ > num_features_tracking_) {
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ > num_features_tracking_bad_) {
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        status_ = FrontendStatus::LOST;
    }
    
    if (tracking_inliers_ < num_features_needed_for_keyframe_) {
        InsertKeyframeAndTriggerBackend();
    }
    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

    if (viewer_) viewer_->AddCurrentFrame(current_frame_);
    return true;
}

// Extract corresponding features from current left frame (to features from last left frame)
int Frontend::FindFeaturesInCurrent() {

    std::vector<cv::Point2f> kps_current;
    std::vector<uchar> status;
    CalcCorrespondingFeatures(last_frame_, current_frame_, camera_left_, 
                              last_frame_->left_img_, current_frame_->left_img_,
                              kps_current, status);

    int num_good_pts = 0;

    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_current[i], 7);
            Feature::Ptr feature(new Feature(current_frame_, kp));
            feature->map_point_ = last_frame_->features_left_[i]->map_point_;  // passing a weak_ptr, maybe nullptr
            current_frame_->features_left_.push_back(feature);
            num_good_pts++;
        }
    }

    // LOG(INFO) << "Find " << num_good_pts << " matched features in the current frame to last frame.";
    return num_good_pts;
}

// Use 3D-2D BA VO to estimate pose of current frame
// 3D points are the points in map coresponding to the left image's feature points
// 2D points are the observed pixels coordinates of left image's feature points
// Initial estimate value is (relative_motion between last 2 frames) * pose of last frame
// Final estimate value is the T_{cw}
int Frontend::EstimateCurrentPose() {
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // pose vertex
    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(current_frame_->Pose());
    optimizer.addVertex(vertex_pose);

    // K
    Mat33 K = camera_left_->K();

    // ID of edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<Feature::Ptr> features;

    // Use the features's connected mappoints (3D) and the features (2D)
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        // This map_point is passed from last frame
        // If this map_point exists, it could serves as 3D-2D points
        auto mp = current_frame_->features_left_[i]->map_point_.lock();    
        if (mp) {
            features.push_back(current_frame_->features_left_[i]);
            EdgeProjectionPoseOnly *edge =
                new EdgeProjectionPoseOnly(mp->GetPos(), K);
            edge->setId(index++);
            edge->setVertex(0, vertex_pose);
            edge->setMeasurement(
                toVec2(current_frame_->features_left_[i]->position_.pt));
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);
            edges.push_back(edge);
            optimizer.addEdge(edge);
        }
    }

    // estimate the Pose the determine the outliers
    const double chi2_th = 5.991;
    int cnt_outlier = 0;

    // Optimize 4 * 10 times
    // During the first 2 * 10 times, set the outlier not optmized
    // During the last 2 * 10 times, remove the kernel
    for (int iteration = 0; iteration < 4; ++iteration) {
        vertex_pose->setEstimate(current_frame_->Pose());

        // Only optmize level 0 edges
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outlier = 0;

        // count the outliers
        for (size_t i = 0; i < edges.size(); ++i) {
            auto e = edges[i];

            // Since the is_outlier_ is true, this feature is not optimized
            // computeError needs to be trigger manually
            if (features[i]->is_outlier_) {
                e->computeError();
            }

            if (e->chi2() > chi2_th) {
                features[i]->is_outlier_ = true;
                e->setLevel(1);
                cnt_outlier++;
            } else {
                features[i]->is_outlier_ = false;
                e->setLevel(0);
            };

            if (iteration == 2) {
                e->setRobustKernel(nullptr);
            }
        }
    }

    // Set pose
    current_frame_->SetPose(vertex_pose->estimate());

    // Reset the assigned mappoint for outliers
    for (auto &feat : features) {
        if (feat->is_outlier_) {
            // Since thie feature hasn't been added to map_point's observation, the observation doesn't need to be changed
            feat->map_point_.reset();

            // Although this feature is mismatched with previous point, it still could assign to a new mappoint with triangulation
            feat->is_outlier_ = false; 
        }
    }
    return features.size() - cnt_outlier;
}

void Frontend::InsertKeyframeAndTriggerBackend() {

    // LOG(INFO) << "Features are not enough for tracking\n. Set frame " << current_frame_->id_ << " as new keyframe with ID" << current_frame_->keyframe_id_;

    current_frame_->SetKeyFrame();

    // detect new features and assign new mappoint for the keyframe
    DetectFeatures();  
    FindFeaturesInRight();
    TriangulateNewPoints();

    // When the keyframe is ready, insert this new keyframe to map_
    map_->InsertKeyFrame(current_frame_);

    // update backend because we have a new keyframe
    backend_->UpdateMap(current_frame_);

    if (viewer_) viewer_->UpdateMap();
}

// Extract ORB features from current left frame 
int Frontend::DetectFeatures() {
    // Mask current features to extract new features
    cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
    for (auto &feat : current_frame_->features_left_) {
        cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                      feat->position_.pt + cv::Point2f(10, 10), 0, CV_FILLED);
    }

    std::vector<cv::KeyPoint> keypoints;
    // gftt_->detect(current_frame_->left_img_, keypoints, mask);
    orb_->detect(current_frame_->left_img_, keypoints, mask);
    int cnt_detected = 0;
    for (auto &kp : keypoints) {
        // append to previous features_left_
        current_frame_->features_left_.push_back(
            Feature::Ptr(new Feature(current_frame_, kp)));
        cnt_detected++;
    }

    // LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}

// Find corresponding features from current right frame (to current left features)
int Frontend::FindFeaturesInRight() {
    
    std::vector<cv::Point2f> kps_right;
    std::vector<uchar> status;
    CalcCorrespondingFeatures(current_frame_, current_frame_, camera_right_, 
                              current_frame_->left_img_, current_frame_->right_img_,
                              kps_right, status);


    int num_good_pts = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_right[i], 7);
            Feature::Ptr feature(new Feature(current_frame_, kp));
            feature->is_on_left_image_ = false;
            current_frame_->features_right_.push_back(feature);
            num_good_pts++;
        } else {
            current_frame_->features_right_.push_back(nullptr);
        }
    }
    // LOG(INFO) << "Find " << num_good_pts << " matched features in the right frame to left frame.";
    return num_good_pts;
}


int Frontend::TriangulateNewPoints() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    SE3 current_pose_Twc = current_frame_->Pose().inverse();
    int cnt_triangulated_pts = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_left_[i]->map_point_.expired() &&
            current_frame_->features_right_[i] != nullptr) {
            // 左图的特征点未关联地图点且存在右图匹配点，尝试三角化
            // The reason the feature_left doesn't have a mappoint could because it doesn't get a match from last frame, or it is set outlier during this time tracking
            std::vector<Vec3> points{
                camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                         current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                    Vec2(current_frame_->features_right_[i]->position_.pt.x,
                         current_frame_->features_right_[i]->position_.pt.y))};
            Vec3 pworld = Vec3::Zero();

            if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                auto new_map_point = MapPoint::CreateNewMappoint();
                pworld = current_pose_Twc * pworld;
                new_map_point->SetPos(pworld);
                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;
                map_->InsertMapPoint(new_map_point);
                cnt_triangulated_pts++;
            }
        }
    }
    // LOG(INFO) << "Triangulate to generate new landmarks: " << cnt_triangulated_pts;
    return cnt_triangulated_pts;
}

// Initialize the map with first left and right frames
bool Frontend::StereoInit() {
    // LOG(INFO) << "Initializing the map.";

    DetectFeatures();
    int num_coor_features = FindFeaturesInRight();
    if (num_coor_features < num_features_init_) {
        return false;
    }

    bool build_map_success = BuildInitMap();
    if (build_map_success) {
        status_ = FrontendStatus::TRACKING_GOOD;
        if (viewer_) {
            viewer_->AddCurrentFrame(current_frame_);
            viewer_->UpdateMap();
        }
        return true;
    }
    return false;
}

void Frontend::CalcCorrespondingFeatures(const Frame::Ptr frame1, const Frame::Ptr frame2, 
                        const Camera::Ptr camera, const Mat &img1, const Mat &img2,
                        std::vector<cv::Point2f> &kps2, std::vector<uchar> &status)
{
    std::vector<cv::Point2f> kps1;

    for(auto &kp: frame1->features_left_)
    {
        kps1.push_back(kp->position_.pt);
        if (kp->map_point_.lock())
        {
            // If the previous feature point is associated with a mappoint
            // use project point from mappoint
            auto mp = kp->map_point_.lock();
            auto px = camera->world2pixel(mp->GetPos(), frame2->Pose());
            kps2.push_back(cv::Point2f(px[0], px[1]));
        }
        else
        {
            // The previous feature point is not associated with a mappoint 
            // (not matched with previous right frame or this frame is a new keyframe)
            // use same pixel in left image
            kps2.push_back(kp->position_.pt);
        }
    }

    Mat error;
    cv::calcOpticalFlowPyrLK(
        img1, img2, kps1, kps2, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);
}

// Triangulate feature points from first left and right frames and build initial map
bool Frontend::BuildInitMap() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    size_t cnt_init_landmarks = 0;
    // Look for the features pairs that both appears in the left and right frames
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_right_[i] == nullptr) continue;
        // create map point from triangulate
        std::vector<Vec3> points{
            camera_left_->pixel2camera(
                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                     current_frame_->features_left_[i]->position_.pt.y)),
            camera_right_->pixel2camera(
                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                     current_frame_->features_right_[i]->position_.pt.y))};
        Vec3 pworld = Vec3::Zero();

        if (triangulation(poses, points, pworld) && pworld[2] > 0) {
            auto new_map_point = MapPoint::CreateNewMappoint();
            new_map_point->SetPos(pworld);
            current_frame_->features_left_[i]->map_point_ = new_map_point;
            cnt_init_landmarks++;
            map_->InsertMapPoint(new_map_point);
        }
    }
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);
    backend_->UpdateMap(current_frame_);

    // LOG(INFO) << "Initial map created with " << cnt_init_landmarks
            //   << " map points";

    return true;
}

bool Frontend::Reset() {
    // LOG(INFO) << "Reset is not implemented. ";
    return true;
}

}  // namespace myslam