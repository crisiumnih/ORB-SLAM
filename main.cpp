#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/tracker/tracker.h" // Adjust path as needed based on your includes

int main() {
    std::string imagePath = "../dataset/V1/0001.png";

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image at " << imagePath << std::endl;
        return -1;
    }

    Tracker tracker;

    std::vector<cv::KeyPoint> keypoints = tracker.homogeneousOrbExtraction(image, 2000);

    std::cout << "Detected " << keypoints.size() << " homogeneous ORB keypoints." << std::endl;

    cv::Mat imgKeypoints;
    cv::drawKeypoints(image, keypoints, imgKeypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    cv::imshow("Homogeneous ORB Keypoints", imgKeypoints);
    cv::waitKey(0);

    return 0;
}