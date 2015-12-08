/**
 This file is part of Image Alignment.
 
 Copyright Christoph Heindl 2015
 
 Image Alignment is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 Image Alignment is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with Image Alignment.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <imagealign/imagealign.h>
IA_DISABLE_PRAGMA_WARN(4190)
IA_DISABLE_PRAGMA_WARN(4244)
#include <opencv2/opencv.hpp>
IA_DISABLE_PRAGMA_WARN_END
IA_DISABLE_PRAGMA_WARN_END
#include <iomanip>
#include <iostream>

/**
    This example is based on OpenCVs Lucas Kanade Optical Flow example. 
    It demonstrates how Image Alignment can be used to perform optical flow.
 */

const int MAX_FEATURES = 500;


void opticalFlowIA(cv::Mat &prevGray,
                   cv::Mat &gray,
                   std::vector<cv::Point2f> &prevPoints,
                   std::vector<cv::Point2f> &points,
                   std::vector<uchar> &status,
                   std::vector<float> &err)
{
    namespace ia = imagealign;
    
    // Will be using pure translational motion
    typedef ia::WarpTranslationF WarpType;
    
    // In conjunction with inverse compositional algorithm
    typedef ia::AlignInverseCompositional< WarpType > AlignType;
    
    // We will also make use of the face, that we can share gray among all aligners
    ia::ImagePyramid target;
    target.create(gray, 3);
    
    // Create an aligner for each point
    std::vector<AlignType> aligners(prevPoints.size());
    
    // Create a warp for each point. Note we use identity transform here.
    std::vector<WarpType> warps(prevPoints.size());
    
    // Prepare outputs
    points.resize(prevPoints.size());
    status.resize(prevPoints.size());
    err.resize(prevPoints.size());
    
    for (size_t i = 0; i < aligners.size(); ++i) {
        
        // The template will be a rectangular region around the point
        const int windowOff = 15;
        const cv::Point2f p = prevPoints[i];
        
        int l = (int)(p.x - windowOff);
        int t = (int)(p.y - windowOff);
        int r = (int)(p.x + windowOff);
        int b = (int)(p.y + windowOff);
        
        // Clamp to region
        l = std::max<int>(0, l);
        t = std::max<int>(0, t);
        r = std::min<int>(gray.cols - 1, r);
        b = std::min<int>(gray.rows - 1, b);
        cv::Rect roi(l, t, r - l, b - t);
        
        // Move corner to top left
        float offsetX = l - p.x;
        float offsetY = t - p.y;
        
        // Initialize warp
        ia::WarpTranslationF::Traits::ParamType wp(p.x + offsetX, p.y + offsetY);
        std::cout << wp.t() << std::endl;
        warps[i].setParameters(wp);
        
        // Initialize aligner
        aligners[i].prepare(prevGray(roi), target, warps[i], 3);
        
        // Align
        int maxIterationsPerLevel[] = {10, 10, 10};
        aligners[i].align(warps[i], maxIterationsPerLevel);
        
        // Extract result
        wp = warps[i].parameters();
        std::cout << wp.t() << std::endl;
        std::cout << "-----" << std::endl;
        points[i].x = wp(0) - offsetX;
        points[i].y = wp(1) - offsetY;
        err[i] = aligners[i].lastError();
        status[i] = aligners[i].lastError() < 100.f ? 255 : 0;
    }
    
}

void opticalFlowCV(cv::Mat &prevGray,
                   cv::Mat &gray,
                   std::vector<cv::Point2f> &prevPoints,
                   std::vector<cv::Point2f> &points,
                   std::vector<uchar> &status,
                   std::vector<float> &err)
{
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
    cv::Size subPixWinSize(10,10), winSize(31,31);
    
    calcOpticalFlowPyrLK(prevGray, gray, prevPoints, points, status, err, winSize, 3, termcrit, 0, 0.001);
}

void drawOpticalFlow(cv::Mat &image,
                     std::vector<cv::Point2f> &prevPoints,
                     std::vector<cv::Point2f> &points,
                     std::vector<uchar> &status)
{
    for (size_t i = 0; i < prevPoints.size(); ++i) {
        if (!status[i])
            continue;
        
        cv::circle(image, points[i], 3, cv::Scalar(0,255,0), -1, 8);
    }
}


int main(int argc, char **argv)
{
    cv::VideoCapture cap;
    
    if (argc > 1) {
        if (isdigit(argv[1][0])) {
            // Open capture device by index
            cap.open(atoi(argv[1]));
        } else {
            // Open video video
            cap.open(argv[1]);
        }
    } else {
        // Open default device;
        cap.open(0);
    }
    
    if (!cap.isOpened()) {
        std::cerr << "Failed to open capture device." << std::endl;
        return -1;
    }
    
    cv::Mat gray, prevGray, image, frame;
    std::vector<cv::Point2f> points[2];
    
    bool init = false;
    bool done = false;
    
    while (!done) {
        
        // Grab frame
        cap >> frame;
        if (frame.empty())
            break;
        
        frame.copyTo(image);
        cv::cvtColor(image, gray, CV_BGR2GRAY);
        
        if (init) {
            cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
            cv::Size subPixWinSize(10,10), winSize(31,31);
            
            // Shi-Thomasi corner features.
            cv::goodFeaturesToTrack(gray, points[1], MAX_FEATURES, 0.01, 10, cv::Mat(), 3, 0, 0.04);
            cv::cornerSubPix(gray, points[1], subPixWinSize, cv::Size(-1,-1), termcrit);
            
            init = false;
            
        } else if (!points[0].empty()) {
            if(prevGray.empty())
                gray.copyTo(prevGray);
            
            std::vector<uchar> status;
            std::vector<float> err;
            
            // Perform optical flow
            opticalFlowIA(prevGray, gray, points[0], points[1], status, err);
            //opticalFlowCV(prevGray, gray, points[0], points[1], status, err);
            drawOpticalFlow(image, points[0], points[1], status);
            
            // Draw optical flow results
            size_t k = 0;
            for (size_t i = 0; i < points[1].size(); ++i) {
                if (!status[i])
                    continue;
                
                points[1][k++] = points[1][i];
            }
            points[1].resize(k);
            
        }
        
        cv::imshow("Optical Flow", image);
        int key = cv::waitKey(10);
        
        switch (key) {
            case 'x':
                done = true;
                break;
                
            case 'r':
                init = true;
                break;
        }
        
        std::swap(points[1], points[0]);
        cv::swap(prevGray, gray);
        
    }
    
    
    return 0;
}




