//
// Created by gaoxiang on 19-5-4.
//

#include "myslam/visual_odometry.h"

int main(int argc, char **argv) {

    if ( argc != 2 )
    {
        std::cout << "usage: run_vo parameter_file" << std::endl;
        return 1;
    }

    myslam::VisualOdometry::Ptr vo(new myslam::VisualOdometry( argv[1] ));
    assert(vo->Init() == true);
    vo->Run();

    return 0;
}
