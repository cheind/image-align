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

namespace ia = imagealign;

const int MAX_FEATURES = 20;

template<class Scalar>
cv::Point_<Scalar> toP(const cv::Matx<Scalar, 2, 1> &p) {
    return cv::Point_<Scalar>(p(0), p(1));
}

template<int WarpType, class Scalar>
void drawRectOfTemplate(cv::Mat &img, const ia::Warp<WarpType, Scalar> &w, cv::Size tplSize, cv::Scalar color)
{
    typedef typename ia::WarpTraits<WarpType, Scalar>::PointType PointType;

    PointType c0 = w(PointType(Scalar(0.5), Scalar(0.5)));
    PointType c1 = w(PointType(Scalar(0.5) + tplSize.width, Scalar(0.5)));
    PointType c2 = w(PointType(Scalar(0.5) + tplSize.width, Scalar(0.5) + tplSize.height));
    PointType c3 = w(PointType(Scalar(0.5), Scalar(0.5) + tplSize.height));

    cv::line(img, toP(c0), toP(c1), color, 1, CV_AA);
    cv::line(img, toP(c1), toP(c2), color, 1, CV_AA);
    cv::line(img, toP(c2), toP(c3), color, 1, CV_AA);
    cv::line(img, toP(c3), toP(c0), color, 1, CV_AA);
}

void opticalFlowIA(cv::Mat &prevGray,
                   cv::Mat &gray,
                   std::vector<cv::Point2f> &prevPoints,
                   std::vector<cv::Point2f> &points,
                   std::vector<uchar> &status,
                   std::vector<float> &err)
{
    const int LEVELS = 3;

    // Will be using pure translational motion
    typedef ia::WarpTranslationF WarpType;
    
    // In conjunction with inverse compositional algorithm
    typedef ia::AlignInverseCompositional< WarpType > AlignType;
    
    // We will also make use of the face, that we can share gray among all aligners
    ia::ImagePyramid target;
    target.create(gray, LEVELS);
    
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
        l = std::min<int>(gray.cols - 1, std::max<int>(0, l));
        t = std::min<int>(gray.rows - 1, std::max<int>(0, t));
        r = std::min<int>(gray.cols - 1, std::max<int>(0, r));
        b = std::min<int>(gray.rows - 1, std::max<int>(0, b));
        cv::Rect roi(l, t, r - l, b - t);

        if (roi.area() == 0) {
            status[i] = 0;
            continue;
        }
        
        // Move corner to top left
        float offsetX = (float)l - p.x;
        float offsetY = (float)t - p.y;
        
        // Initialize warp
        ia::WarpTranslationF::Traits::ParamType wp(p.x + offsetX, p.y + offsetY);
        warps[i].setParameters(wp);

        /*
        cv::imwrite("target.png", gray);
        cv::imwrite("template.png", prevGray(roi));
        std::cout << wp << std::endl;
        std::cout << "----" << std::endl;
        */

        /*
        cv::imshow("roi", prevGray(roi));
        cv::Mat tmp, tmp2;
        cv::cvtColor(gray, tmp, CV_GRAY2BGR);
        */

        // Initialize aligner
        aligners[i].prepare(prevGray(roi), target, warps[i], LEVELS);
        
        // Align
        int maxIterationsPerLevel[LEVELS] = {10, 10, 10};
        //aligners[i].align(warps[i], maxIterationsPerLevel);
        for (int l = 0; l < LEVELS; ++l) {
            aligners[i].setLevel(l);
            for (int iter = 0; iter < maxIterationsPerLevel[l]; ++iter) {
                aligners[i].align(warps[i]);
                /*

                wp = warps[i].parameters();
                
                tmp.copyTo(tmp2);
                drawRectOfTemplate(tmp2, warps[i], roi.size(), cv::Scalar(0, 255, 0));
                cv::imshow("target", tmp2);
                cv::waitKey();
                */
            }
        }


        
        // Extract result
        wp = warps[i].parameters();
        points[i].x = wp(0) - offsetX;
        points[i].y = wp(1) - offsetY;
        err[i] = aligners[i].lastError();
        status[i] = 255; //aligners[i].lastError() < 5000.f ? 255 : 0;

       // drawRectOfTemplate(tmp, warps[i], roi.size(), cv::Scalar(0, 0, 255));

    }

    std::cout << "frame" << std::endl;
    
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




