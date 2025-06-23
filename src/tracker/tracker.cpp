#include "tracker.h"
#include <opencv2/features2d.hpp>

Tracker::Tracker() : n_features_(1500), n_levels_(8), scale_factor_(1.2f), grid_rows_(8), grid_cols_(8), min_corners_per_cell_(5) {}

void Tracker::processFrame(const cv::Mat& frame, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    // Convert to grayscale if needed
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame.clone();
    }

    std::vector<cv::Mat> pyramid;
    createImagePyramid(gray, pyramid);

    keypoints.clear();
    distributeFeatures(pyramid, keypoints);

    computeORBDescriptors(gray, keypoints, descriptors);
}

void Tracker::createImagePyramid(const cv::Mat& frame, std::vector<cv::Mat>& pyramid) {
    pyramid.resize(n_levels_);
    pyramid[0] = frame.clone();
    for (int i = 1; i < n_levels_; ++i) {
        cv::resize(pyramid[i-1], pyramid[i], cv::Size(), 1.0/scale_factor_, 1.0/scale_factor_, cv::INTER_LINEAR);
    }
}

void Tracker::detectFastCornersInCell(const cv::Mat& image, int row_start, int row_end, int col_start, int col_end,
                                      std::vector<cv::KeyPoint>& keypoints, int& threshold) {
    std::vector<cv::KeyPoint> cell_keypoints;
    cv::FAST(image(cv::Rect(col_start, row_start, col_end - col_start, row_end - row_start)), 
             cell_keypoints, threshold, true);

    if (cell_keypoints.size() < min_corners_per_cell_ && threshold > 5) {
        threshold = std::max(5, threshold - 5);
        detectFastCornersInCell(image, row_start, row_end, col_start, col_end, keypoints, threshold);
        return;
    } else if (cell_keypoints.size() > min_corners_per_cell_ * 2 && threshold < 50) {
        threshold += 5;
        detectFastCornersInCell(image, row_start, row_end, col_start, col_end, keypoints, threshold);
        return;
    }

    // Adjust keypoint coordinates to full image
    for (auto& kp : cell_keypoints) {
        kp.pt.x += col_start;
        kp.pt.y += row_start;
        keypoints.push_back(kp);
    }
}

void Tracker::distributeFeatures(const std::vector<cv::Mat>& pyramid, std::vector<cv::KeyPoint>& keypoints) {
    for (int level = 0; level < n_levels_; ++level) {
        const cv::Mat& image = pyramid[level];
        float scale = std::pow(scale_factor_, level);
        int cell_width = image.cols / grid_cols_;
        int cell_height = image.rows / grid_rows_;

        for (int i = 0; i < grid_rows_; ++i) {
            for (int j = 0; j < grid_cols_; ++j) {
                int row_start = i * cell_height;
                int row_end = (i + 1) * cell_height;
                int col_start = j * cell_width;
                int col_end = (j + 1) * cell_width;

                // Ensure boundaries
                row_end = std::min(row_end, image.rows);
                col_end = std::min(col_end, image.cols);

                std::vector<cv::KeyPoint> cell_keypoints;
                int threshold = 20; // Initial FAST threshold
                detectFastCornersInCell(image, row_start, row_end, col_start, col_end, cell_keypoints, threshold);

                // Scale keypoints to original image size
                for (auto& kp : cell_keypoints) {
                    kp.pt.x *= scale;
                    kp.pt.y *= scale;
                    kp.size *= scale;
                    kp.octave = level;
                    keypoints.push_back(kp);
                }
            }
        }
    }

    // Limit total keypoints
    if (keypoints.size() > n_features_) {
        std::sort(keypoints.begin(), keypoints.end(), 
                  [](const cv::KeyPoint& a, const cv::KeyPoint& b) { return a.response > b.response; });
        keypoints.resize(n_features_);
    }
}

void Tracker::computeORBDescriptors(const cv::Mat& frame, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->compute(frame, keypoints, descriptors);
}