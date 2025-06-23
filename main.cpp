#include <opencv2/opencv.hpp>
#include <iostream>
#include "tracker.h"

int main() {
    cv::Mat frame = cv::imread("dataset/hotel/0001.png");
    if (frame.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    Tracker tracker;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    tracker.processFrame(frame, keypoints, descriptors);

    cv::Mat output;
    cv::drawKeypoints(frame, keypoints, output, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("ORB Features", output);
    cv::waitKey(0);

    std::cout << "Extracted " << keypoints.size() << " keypoints" << std::endl;

    return 0;
}