#include "myslam/loopclosing.h"

namespace myslam {

void LoopClosing::Run()
{
    while (loopclosing_running_.load())
    {
        std::unique_lock<std::mutex> lock(data_mutex_);
        // Stuck current thread until the map_update_ is notified by other thread 
        // This automatically unlock the data_mutex_
        map_update_.wait(lock);
        LOG(INFO) << "Start detecting loop\n";
    }
}

} // namespace