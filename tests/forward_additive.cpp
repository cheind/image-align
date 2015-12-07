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
#include <imagealign/warp_image.h>
#include <opencv2/opencv.hpp>

TEST_CASE("forward-additive")
{
    namespace ia = imagealign;
    
    typedef ia::Warp<ia::WARP_TRANSLATION> WarpType;
    typedef ia::WarpTraits<ia::WARP_TRANSLATION> Traits;
    typedef ia::AlignForwardAdditive<ia::WARP_TRANSLATION> AlignType;
    
    cv::Mat target(100, 100, CV_8UC1);
    cv::randu(target, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::blur(target, target, cv::Size(5,5));
    
    cv::Mat tmpl = target(cv::Rect(20, 20, 10, 10));
    
    AlignType al;
    al.prepare(tmpl, target, 1);
    
    WarpType w;
    w.setParameters(Traits::ParamType(15, 15));
    
    al.align(w, 100, 0.001f);
    
    REQUIRE(al.iteration() < 100);
    REQUIRE(w.parameters()(0,0) == Catch::Detail::Approx(20).epsilon(0.01));
    REQUIRE(w.parameters()(1,0) == Catch::Detail::Approx(20).epsilon(0.01));
}

TEST_CASE("forward-additive-euclidean")
{
    namespace ia = imagealign;
    
    typedef ia::Warp<ia::WARP_EUCLIDEAN> WarpType;
    typedef ia::WarpTraits<ia::WARP_EUCLIDEAN> Traits;
    typedef ia::AlignForwardAdditive<ia::WARP_EUCLIDEAN> AlignType;
    
    cv::Mat target(100, 100, CV_8UC1);
    cv::randu(target, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::blur(target, target, cv::Size(5,5));
    
    cv::Mat tmpl = cv::Mat(20, 20, CV_8UC1);
    
    Traits::ParamType real(cv::theRNG().uniform(20.f, 40.f),
                        cv::theRNG().uniform(20.f, 40.f),
                        cv::theRNG().uniform(0.17f, 0.78f));
    
    WarpType w;
    w.setParameters(real);
    
    ia::warpImage<uchar, ia::SAMPLE_BILINEAR>(target, tmpl, tmpl.size(), w);
    
    // Perturbate warp
    Traits::ParamType r((float)cv::theRNG().gaussian(2.f),
                        (float)cv::theRNG().gaussian(2.f),
                        (float)cv::theRNG().gaussian(0.5f));
    
    w.setParameters(real + r);
    
    AlignType al;
    al.prepare(tmpl, target, 1);
    al.align(w, 100, 0.001f);
    
    REQUIRE(al.iteration() < 100);
    REQUIRE(w.parameters()(0,0) == Catch::Detail::Approx(real(0,0)).epsilon(0.01));
    REQUIRE(w.parameters()(1,0) == Catch::Detail::Approx(real(1,0)).epsilon(0.01));
    REQUIRE(w.parameters()(2,0) == Catch::Detail::Approx(real(2,0)).epsilon(0.01));

}

TEST_CASE("forward-additive-similarity")
{
    namespace ia = imagealign;
    
    typedef ia::Warp<ia::WARP_SIMILARITY> WarpType;
    typedef ia::WarpTraits<ia::WARP_SIMILARITY> Traits;
    typedef ia::AlignForwardAdditive<ia::WARP_SIMILARITY> AlignType;
    
    cv::Mat target(100, 100, CV_8UC1);
    cv::randu(target, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::blur(target, target, cv::Size(5,5));
    
    cv::Mat tmpl = cv::Mat(20, 20, CV_8UC1);
    
    Traits::ParamType real;
    real << 29.110104f, 20.72f, 0.66f, 1.094f;
    
    WarpType w;
    w.setParametersInCanonicalRepresentation(real);
    
    Traits::ParamType realp = w.parameters();
    
    ia::warpImage<uchar, ia::SAMPLE_BILINEAR>(target, tmpl, tmpl.size(), w);
    
    // Perturbate warp
    Traits::ParamType r;
    r << 0.74045879f, 0.92933869f, -0.21553245f, 0.16472194f;
    
    w.setParametersInCanonicalRepresentation(real + r);
    
    AlignType al;
    al.prepare(tmpl, target, 2);
    const int iterations[] = { 50, 50 };
    al.align(w, iterations);
    
    REQUIRE(al.iteration() < 100);
    REQUIRE(w.parameters()(0,0) == Catch::Detail::Approx(realp(0,0)).epsilon(0.01));
    REQUIRE(w.parameters()(1,0) == Catch::Detail::Approx(realp(1,0)).epsilon(0.01));
    REQUIRE(w.parameters()(2,0) == Catch::Detail::Approx(realp(2,0)).epsilon(0.01));
    REQUIRE(w.parameters()(3,0) == Catch::Detail::Approx(realp(3,0)).epsilon(0.01));
    
}