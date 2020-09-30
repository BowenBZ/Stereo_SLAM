#ifndef MYSLAM_LOOPCLOSING_H
#define MYSLAM_LOOPCLOSING_H

#include "myslam/common_include.h"

namespace myslam {

class LoopClosing {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<LoopClosing> Ptr;

    // Start the loop closure detection thread and keep it
    LoopClosing() {
        loopclosing_running_.store(true);
        loopclosing_thread_ = std::thread(std::bind(&LoopClosing::Run, this));
    }

    // Start the detection once
    void DetectLoop() {
        std::unique_lock<std::mutex> lock(data_mutex_);
        map_update_.notify_one();
    }

    // Stop the loop closure detection thread
    void Stop() {
        loopclosing_running_.store(false);
        loopclosing_thread_.join();
    }

    // The thread of loop closure detection
    std::thread loopclosing_thread_;

private:
    // The main function of loop closure detection
    void Run();

    // The flag to indicate whether the thread is running
    std::atomic<bool> loopclosing_running_;

    // Lock of data
    std::mutex data_mutex_;

    // The variable to trigger detect once
    std::condition_variable map_update_;

};

} // namespace

#endif // LOOPCLOSING_H