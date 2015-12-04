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

#include "catch.hpp"

#include <imagealign/forward_additive.h>
#include <opencv2/opencv.hpp>

TEST_CASE("forward-additive")
{
    namespace ia = imagealign;
    
    typedef ia::Warp<ia::WARP_TRANSLATION> WarpType;
    typedef ia::AlignForwardAdditive<ia::WARP_TRANSLATION> AlignType;
    
    cv::Mat target(100, 100, CV_8UC1);
    cv::randu(target, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::blur(target, target, cv::Size(5,5));
    
    cv::Mat tmpl = target(cv::Rect(20, 20, 10, 10));
    
    AlignType al;
    al.prepare(tmpl, target);
    
    WarpType w;
    w.setParameters(WarpType::VType(15, 15));
    
    int iter = 0;
    while (al.align(w) > 0.001f && iter < 100)
        ++iter;
    
    REQUIRE(iter < 100);
    REQUIRE(w.getParameters()(0,0) == Catch::Detail::Approx(20).epsilon(0.01));
    REQUIRE(w.getParameters()(1,0) == Catch::Detail::Approx(20).epsilon(0.01));
}