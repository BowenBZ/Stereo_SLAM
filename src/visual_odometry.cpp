//
// Created by gaoxiang on 19-5-4.
//
#include "myslam/visual_odometry.h"
#include <chrono>
#include "myslam/config.h"
#include "DBoW3/DBoW3.h"

namespace myslam {

VisualOdometry::VisualOdometry(const std::string &config_path)
    : config_file_path_(config_path) {}

bool VisualOdometry::Init() {
    // read from config file
    if (Config::SetParameterFile(config_file_path_) == false) {
        return false;
    }

    dataset_ =
        Dataset::Ptr(new Dataset(Config::Get<std::string>("dataset_dir")));
    CHECK_EQ(dataset_->Init(), true);

    // Load ORB vocabulary
    LOG(INFO) << "Loading ORB Vocabulary...";
    DBoW3::Vocabulary* mpVocabulary = new DBoW3::Vocabulary(Config::Get<std::string>("vocabulary_dir"));
    LOG(INFO) << "Vocabulary loaded!";

    // create components and links
    frontend_ = Frontend::Ptr(new Frontend);
    backend_ = Backend::Ptr(new Backend);
    map_ = Map::Ptr(new Map);
    viewer_ = Viewer::Ptr(new Viewer);
    loopclosing_ = LoopClosing::Ptr(new LoopClosing(mpVocabulary));

    frontend_->SetBackend(backend_);
    frontend_->SetMap(map_);
    frontend_->SetViewer(viewer_);
    frontend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));
    frontend_->SetLoopClosing(loopclosing_);

    backend_->SetMap(map_);
    backend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));

    viewer_->SetMap(map_);

    loopclosing_->SetORBExtractor(frontend_->GetORBExtractor());
    loopclosing_->SetMap(map_);


    return true;
}

void VisualOdometry::Run() {
    while (1) {
        // LOG(INFO) << "VO is running";
        if (Step() == false) {
            break;
        }
    }

    backend_->Stop();
    viewer_->Close();

    // LOG(INFO) << "VO exit";
}

bool VisualOdometry::Step() {
    Frame::Ptr new_frame = dataset_->NextFrame();
    if (new_frame == nullptr) return false;

    auto t1 = std::chrono::steady_clock::now();
    bool success = frontend_->AddFrame(new_frame);
    auto t2 = std::chrono::steady_clock::now();
    auto time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    // LOG(INFO) << "VO cost time: " << time_used.count() << " seconds.";
    // std::cout << std::endl;
    return success;
}

}  // namespace myslam
