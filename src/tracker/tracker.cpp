#include "tracker.h"
#include <iostream>
#include <algorithm> // For std::sort

Tracker::Tracker() {
    // Constructor if needed
}

std::vector<cv::KeyPoint> Tracker::homogeneousOrbExtraction(const cv::Mat& image, int nfeatures,
                                                           float scaleFactor, int nlevels,
                                                           int edgeThreshold, int firstLevel,
                                                           int WTA_K, cv::ORB::ScoreType scoreType,
                                                           int patchSize, int fastThreshold) {
    std::vector<cv::KeyPoint> allKeypoints;

    // Create an ORB detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create(nfeatures * 2, // Temporarily request more features
                                           scaleFactor,
                                           nlevels,
                                           edgeThreshold,
                                           firstLevel,
                                           WTA_K,
                                           scoreType,
                                           patchSize,
                                           fastThreshold);

    // Detect ORB features
    orb->detect(image, allKeypoints);

    // Apply homogeneous distribution
    int gridRows = 30;
    int gridCols = 30;
    int minCornersPerCell = 5;

    distributeKeypoints(allKeypoints, image.cols, image.rows, gridRows, gridCols, minCornersPerCell);

    if (allKeypoints.size() > nfeatures) {
        std::sort(allKeypoints.begin(), allKeypoints.end(),
                  [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                      return a.response > b.response;
                  });
        allKeypoints.erase(allKeypoints.begin() + nfeatures, allKeypoints.end());
    }

    return allKeypoints;
}

void Tracker::distributeKeypoints(std::vector<cv::KeyPoint>& keypoints, int imageWidth, int imageHeight,
                                 int gridRows, int gridCols, int minCornersPerCell) {
    std::vector<std::vector<cv::KeyPoint>> gridCells(gridRows * gridCols);

    float cellWidth = static_cast<float>(imageWidth) / gridCols;
    float cellHeight = static_cast<float>(imageHeight) / gridRows;

    for (const auto& kp : keypoints) {
        int col = static_cast<int>(kp.pt.x / cellWidth);
        int row = static_cast<int>(kp.pt.y / cellHeight);

        col = std::min(col, gridCols - 1);
        row = std::min(row, gridRows - 1);

        gridCells[row * gridCols + col].push_back(kp);
    }

    std::vector<cv::KeyPoint> homogeneouslyDistributedKeypoints;

    for (int i = 0; i < gridRows * gridCols; ++i) {
        std::sort(gridCells[i].begin(), gridCells[i].end(),
                  [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                      return a.response > b.response;
                  });

        for (int k = 0; k < std::min((int)gridCells[i].size(), minCornersPerCell); ++k) {
            homogeneouslyDistributedKeypoints.push_back(gridCells[i][k]);
        }
    }

    keypoints = homogeneouslyDistributedKeypoints;
}