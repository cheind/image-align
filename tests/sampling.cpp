/**
This file is part of Active Appearance Models (AMM).

Copyright Christoph Heindl 2015
Copyright Sebastian Zambal 2015

AMM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

AMM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with AMM.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "catch.hpp"
#include <imagealign/sampling.h>


TEST_CASE("sampling-bilinear")
{
    namespace ia = imagealign;
    
    ia::Sampler<ia::SAMPLE_BILINEAR> s;
    
    cv::Mat img(2, 2, CV_8UC1);
    
    img.at<uchar>(0,0) = 0;
    img.at<uchar>(0,1) = 64;
    img.at<uchar>(1,0) = 128;
    img.at<uchar>(1,1) = 192;

    typedef cv::Matx<double, 2, 1> PointType;
    
    // Pixel centers
    REQUIRE(s.sample<uchar>(img, PointType(0.0, 0.0)) == 0);
    REQUIRE(s.sample<uchar>(img, PointType(0.0, 1.0)) == 128);
    REQUIRE(s.sample<uchar>(img, PointType(1.0, 0.0)) == 64);
    REQUIRE(s.sample<uchar>(img, PointType(1.0, 1.0)) == 192);

    // Off-centers
    REQUIRE(s.sample<uchar>(img, PointType(0.5, 0.0)) == 32);
    REQUIRE(s.sample<uchar>(img, PointType(0.5, 0.5)) == 96);
}

TEST_CASE("sampling-nearest")
{
    namespace ia = imagealign;
    
    ia::Sampler<ia::SAMPLE_NEAREST> s;
    
    cv::Mat img(2, 2, CV_8UC1);
    
    img.at<uchar>(0,0) = 0;
    img.at<uchar>(0,1) = 64;
    img.at<uchar>(1,0) = 128;
    img.at<uchar>(1,1) = 192;
    
    typedef cv::Matx<double, 2, 1> PointType;
    
    // Pixel centers
    REQUIRE(s.sample<uchar>(img, PointType(0.0, 0.0)) == 0);
    REQUIRE(s.sample<uchar>(img, PointType(0.0, 1.0)) == 128);
    REQUIRE(s.sample<uchar>(img, PointType(1.0, 0.0)) == 64);
    REQUIRE(s.sample<uchar>(img, PointType(1.0, 1.0)) == 192);
    
    // Off-centers
    REQUIRE(s.sample<uchar>(img, PointType(0.5, 0.0)) == 0);
    REQUIRE(s.sample<uchar>(img, PointType(0.5, 0.5)) == 0);
    REQUIRE(s.sample<uchar>(img, PointType(1.1, 0.0)) == 64);

}