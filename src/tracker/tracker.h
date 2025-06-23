#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>
#include <vector>

class Tracker {
public:
    Tracker();

    // Method to perform homogeneous ORB feature extraction
    std::vector<cv::KeyPoint> homogeneousOrbExtraction(const cv::Mat& image, int nfeatures = 2000,
                                                       float scaleFactor = 1.2f, int nlevels = 8,
                                                       int edgeThreshold = 31, int firstLevel = 0,
                                                       int WTA_K = 2, cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE,
                                                       int patchSize = 31, int fastThreshold = 20);

private:
    // Helper function to distribute keypoints homogeneously
    void distributeKeypoints(std::vector<cv::KeyPoint>& keypoints, int imageWidth, int imageHeight,
                             int gridRows, int gridCols, int minCornersPerCell);
};

#endif // TRACKER_H