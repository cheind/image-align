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

namespace ia = imagealign;

void initializeWarp(cv::Size templateSize, cv::Size targetSize, ia::Warp<ia::WARP_TRANSLATION, float> &w) {
    ia::WarpTraits<ia::WARP_TRANSLATION, float>::ParamType params;
    params(0,0) = cv::theRNG().uniform(0.f, (float)(targetSize.width - templateSize.width));
    params(1,0) = cv::theRNG().uniform(0.f, (float)(targetSize.height - templateSize.height));
    
    w.setParameters(params);
}

void perturbateWarp(ia::Warp<ia::WARP_TRANSLATION, float> &w) {
    
    ia::WarpTraits<ia::WARP_TRANSLATION, float>::ParamType params = w.parameters();
    params(0,0) += (float)cv::theRNG().gaussian(8.f);
    params(1,0) += (float)cv::theRNG().gaussian(8.f);
    
    w.setParameters(params);
}

void initializeWarp(cv::Size templateSize, cv::Size targetSize, ia::Warp<ia::WARP_EUCLIDEAN, float> &w) {
    ia::WarpTraits<ia::WARP_EUCLIDEAN, float>::ParamType params;
    params(0,0) = cv::theRNG().uniform(0.f, (float)(targetSize.width - templateSize.width));
    params(1,0) = cv::theRNG().uniform(0.f, (float)(targetSize.height - templateSize.height));
    params(2,0) = cv::theRNG().uniform(0.f, 3.1415f * 0.5f);
    
    w.setParameters(params);
}

void perturbateWarp(ia::Warp<ia::WARP_EUCLIDEAN, float> &w) {
    
    ia::WarpTraits<ia::WARP_EUCLIDEAN, float>::ParamType params = w.parameters();
    params(0,0) += (float)cv::theRNG().gaussian(8.f);
    params(1,0) += (float)cv::theRNG().gaussian(8.f);
    params(2,0) += (float)cv::theRNG().gaussian(0.2f);
    
    w.setParameters(params);
}

void initializeWarp(cv::Size templateSize, cv::Size targetSize, ia::Warp<ia::WARP_SIMILARITY, float> &w) {
    ia::WarpTraits<ia::WARP_SIMILARITY, float>::ParamType params;
    
    params(0,0) = (float)cv::theRNG().uniform(0.f, (float)(targetSize.width - templateSize.width));
    params(1,0) = (float)cv::theRNG().uniform(0.f, (float)(targetSize.height - templateSize.height));
    params(2,0) = (float)cv::theRNG().uniform(0.f, 3.1415f * 0.5f);
    params(3,0) = (float)cv::theRNG().uniform(0.5f, 1.5f);
    
    w.setParametersInCanonicalRepresentation(params);
}

void perturbateWarp(ia::Warp<ia::WARP_SIMILARITY, float> &w) {
    
    // Note parameters are tx, ty, a and b. So we rather use the canoncial form
    ia::WarpTraits<ia::WARP_SIMILARITY, float>::ParamType params = w.parametersInCanonicalRepresentation();
    
    params(0,0) += (float)cv::theRNG().gaussian(3.f);
    params(1,0) += (float)cv::theRNG().gaussian(3.f);
    params(2,0) += (float)cv::theRNG().gaussian(0.2f);
    params(3,0) += (float)cv::theRNG().gaussian(0.05f);
    
    w.setParametersInCanonicalRepresentation(params);
}

template<class Scalar>
cv::Point_<Scalar> toP(const cv::Matx<Scalar, 2, 1> &p) {
    return cv::Point_<Scalar>(p(0), p(1));
}

template<int WarpType>
void drawRectOfTemplate(cv::Mat &img, const ia::Warp<WarpType, float> &w, cv::Size tplSize, cv::Scalar color)
{
    typedef typename ia::WarpTraits<WarpType, float>::PointType PointType;
    
    PointType c0 = w(PointType(0.5f, 0.5f));
    PointType c1 = w(PointType(0.5f + tplSize.width, 0.5f));
    PointType c2 = w(PointType(0.5f + tplSize.width, 0.5f + tplSize.height));
    PointType c3 = w(PointType(0.5f, 0.5f + tplSize.height));
    
    cv::line(img, toP(c0), toP(c1), color, 1, CV_AA);
    cv::line(img, toP(c1), toP(c2), color, 1, CV_AA);
    cv::line(img, toP(c2), toP(c3), color, 1, CV_AA);
    cv::line(img, toP(c3), toP(c0), color, 1, CV_AA);
}

int main(int argc, char **argv)
{
    
    // Choose a warp
    typedef ia::WarpSimilarityF WarpType;
    
    // Choose an alignment strategy
    typedef ia::AlignInverseCompositional<WarpType> AlignType;
    // typedef ia::AlignForwardAdditive<ia::WARP_SIMILARITY> AlignType;
    // typedef ia::AlignForwardCompositional<ia::WARP_SIMILARITY> AlignType;
    
    
    cv::theRNG().state = cv::getTickCount();
    
    cv::Mat target;
    
    if (argc == 1) {
        std::cout << "Generating random image..." << std::endl
                  << "Use " << argv[0] << " <image> to start with a pre-defined image." << std::endl;
        target.create(480, 640, CV_8UC1);
        cv::randu(target, cv::Scalar::all(0), cv::Scalar::all(255));
        cv::blur(target, target, cv::Size(5,5));
    } else {
        target = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    }
    
    cv::Mat tpl(target.size().height / 10, target.size().width / 10, CV_8UC1);
    
    std::cout << "Press 'n' to start a new alignment problem" << std::endl;
    std::cout << "Press 'x' to quit" << std::endl;
    std::cout << "Press any other key to cylce through alignment progress" << std::endl;
    
    bool done = false;
    while (!done) {
        
        // Generate random warp
        WarpType w;
        initializeWarp(tpl.size(), target.size(), w);
        WarpType targetW = w;
        
        // Generate template image from target
        ia::warpImage<uchar, ia::SAMPLE_BILINEAR>(target, tpl, tpl.size(), w);
        
        // Perturbate warp
        perturbateWarp(w);
        
        // Align
        std::vector<WarpType> incrementals;
        incrementals.push_back(w);
        
        const int levels = 3;
        AlignType at;
        at.prepare(tpl, target, levels);
        
        int iterationsPerLevel[] = {30, 30, 15};
        int iterationsPerformed[] = {0, 0, 0};
        
        int64 e1 = cv::getTickCount();
        
        for (int i = 0; i < levels; ++i) {
            at.setLevel(i);
            
            while (iterationsPerformed[i] < iterationsPerLevel[i] &&
                   at.errorChange() > 0.f &&
                   (iterationsPerformed[i] == 0 || cv::norm(at.lastIncrement()) > 0.01f))
            {
                at.align(w);
                incrementals.push_back(w);
                ++iterationsPerformed[i];
            }
            
        }
        
        double elapsed = (cv::getTickCount() - e1) / cv::getTickFrequency();
        
        std::cout << "Completed after [" << iterationsPerformed[0] << "," << iterationsPerformed[1] << "," << iterationsPerformed[2] << "] iterations. "
                  << "Last error: " << at.lastError() << " "
                  << "Took " << elapsed << " seconds." << std::endl;
        
        
        cv::Mat display;
        cv::cvtColor(target, display, CV_GRAY2BGR);
        drawRectOfTemplate(display, targetW, tpl.size(), cv::Scalar(0, 0, 255));
        
        cv::imshow("Template", tpl);
        
        for (size_t i = 0; i < incrementals.size(); ++i) {
            
            cv::Mat warped;
            ia::warpImage<uchar, ia::SAMPLE_BILINEAR>(target, warped, tpl.size(), incrementals[i]);
            cv::imshow("Warped", warped);
            
            cv::Mat dispClone = display.clone();
            drawRectOfTemplate(dispClone, incrementals[i], tpl.size(), cv::Scalar(0, 255, 0));
            
            cv::imshow("Image", dispClone);
            int key = cv::waitKey();
            if (key == 'x') {
                done = true;
                break;
            } else if (key == 'n') {
                break;
            }
        }
    }
    
    return 0;
}




