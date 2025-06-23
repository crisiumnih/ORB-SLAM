#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>
#include <vector>

class Tracker {
public:
    Tracker();
    // Process a single frame to extract homogeneous ORB features
    void processFrame(const cv::Mat& frame, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

private:
    // Parameters for ORB
    int n_features_;
    int n_levels_;   // Number of pyramid levels (8)
    float scale_factor_; // Scale factor between pyramid levels
    int grid_rows_;  // Grid rows for homogeneous distribution
    int grid_cols_;  // Grid columns for homogeneous distribution
    int min_corners_per_cell_; // Minimum corners per grid cell

    // Create image pyramid
    void createImagePyramid(const cv::Mat& frame, std::vector<cv::Mat>& pyramid);
    // Detect FAST corners in a grid cell with threshold
    void detectFastCornersInCell(const cv::Mat& image, int row_start, int row_end, int col_start, int col_end,
                                 std::vector<cv::KeyPoint>& keypoints, int& threshold);
    // Distribute features homogeneously
    void distributeFeatures(const std::vector<cv::Mat>& pyramid, std::vector<cv::KeyPoint>& keypoints);
    // Compute ORB descriptors
    void computeORBDescriptors(const cv::Mat& frame, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
};

#endif // TRACKER_H