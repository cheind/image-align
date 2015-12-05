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
#include <opencv2/opencv.hpp>

namespace ia = imagealign;

void initializeWarp(cv::Size templateSize, cv::Size targetSize, ia::Warp<ia::WARP_TRANSLATION> &w) {
    ia::WarpTraits<ia::WARP_TRANSLATION>::ParamType params;
    params(0,0) = cv::theRNG().uniform(0, targetSize.width - templateSize.width);
    params(1,0) = cv::theRNG().uniform(0, targetSize.height - templateSize.height);
    
    w.setParameters(params);
}

void perturbateWarp(ia::Warp<ia::WARP_TRANSLATION> &w) {
    
    ia::WarpTraits<ia::WARP_TRANSLATION>::ParamType params = w.getParameters();
    params(0,0) += cv::theRNG().gaussian(8);
    params(1,0) += cv::theRNG().gaussian(8);
    
    w.setParameters(params);
}

template<int WarpType>
void drawRectOfTemplate(cv::Mat &img, const ia::Warp<WarpType> &w, cv::Size tplSize, cv::Scalar color) {
    
    cv::Point2f c0 = w(cv::Point2f(0.5f, 0.5f));
    cv::Point2f c1 = w(cv::Point2f(0.5f + tplSize.width, 0.5f));
    cv::Point2f c2 = w(cv::Point2f(0.5f + tplSize.width, 0.5f + tplSize.height));
    cv::Point2f c3 = w(cv::Point2f(0.5f, 0.5f + tplSize.height));
    
    cv::line(img, c0, c1, color, 1, CV_AA);
    cv::line(img, c1, c2, color, 1, CV_AA);
    cv::line(img, c2, c3, color, 1, CV_AA);
    cv::line(img, c3, c0, color, 1, CV_AA);
}

int main(int argc, char **argv)
{
    typedef ia::Warp<ia::WARP_TRANSLATION> WarpType;
    typedef ia::AlignForwardAdditive<ia::WARP_TRANSLATION> AlignType;
    
    cv::theRNG().state = cv::getTickCount();
    
    // Random target image
    cv::Mat target;
    
    if (argc == 1) {
        target.create(480, 640, CV_8UC1);
        cv::randu(target, cv::Scalar::all(0), cv::Scalar::all(255));
        cv::blur(target, target, cv::Size(5,5));
    } else {
        target = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    }
    
    cv::Mat tpl(target.size().height / 10, target.size().width / 10, CV_8UC1);
    
    bool done = false;
    while (!done) {
        // Generate random warp
        WarpType w;
        initializeWarp(tpl.size(), target.size(), w);
        WarpType targetW = w;
        
        std::cout << targetW.getParameters().t() << std::endl;
        
        // Generate template image from target
        ia::warpImage<uchar>(target, tpl, tpl.size(), w);
        
        // Perturbate warp
        perturbateWarp(w);
        
        // Align
        std::vector<WarpType> incrementals;
        incrementals.push_back(w);
        
        AlignType at;
        at.prepare(tpl, target);
        
        int iter = 0;
        while (iter < 100) {
            at.align(w);
            ++iter;
            incrementals.push_back(w);
        }
        
        cv::Mat display;
        cv::cvtColor(target, display, CV_GRAY2BGR);
        drawRectOfTemplate(display, targetW, tpl.size(), cv::Scalar(0, 0, 255));
        
        cv::imshow("Template", tpl);
        
        for (size_t i = 0; i < incrementals.size(); ++i) {
            
            cv::Mat warped;
            ia::warpImage<uchar>(target, warped, tpl.size(), incrementals[i]);
            cv::imshow("Warped", warped);
            
            cv::Mat dispClone = display.clone();
            drawRectOfTemplate(dispClone, incrementals[i], tpl.size(), cv::Scalar(0, 255, 0));
            
            std::cout << incrementals[i].getParameters().t() << std::endl;
            
            cv::imshow("Image", dispClone);
            int key = cv::waitKey();
            if (key == 'x') {
                done = true;
                break;
            } else if (key == 'n') {
                break;
            }
        }
        
        // Visualize
    }
    
    return 0;
}




