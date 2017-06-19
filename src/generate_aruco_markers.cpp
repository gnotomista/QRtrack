#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

int main(int argc, char** argv )
{
    std::ostringstream oss;

    cv::Mat marker_image;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    
    for (int i = 0; i<25; i++) {
        cv::aruco::drawMarker(dictionary, i, 75, marker_image, 1);
        
        oss.str("");
        oss << argv[1] << "/marker_" << i << ".png";
        
        imwrite(oss.str(), marker_image);
    }

    return 0;
}
