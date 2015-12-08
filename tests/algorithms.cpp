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
#include <imagealign/forward_compositional.h>
#include <imagealign/inverse_compositional.h>
#include <imagealign/warp_image.h>
#include <iostream>

template< class A, class W >
W testAlgorithm(cv::Mat tpl, cv::Mat target, W w, int levels, const typename W::Traits::ParamType &expected, double tolerance = 0.01)
{
    typedef typename W::Traits::ScalarType S;
    
    A a;
    a.prepare(tpl, target, w, levels);
    
    for (int i = 0; i < levels; ++i) {
        a.setLevel(i);
        a.align(w, 100 / levels, S(0.001));
    }
    
    REQUIRE(a.iteration() < 100);
    REQUIRE(cv::norm(w.parameters() - expected, cv::NORM_L1) == Catch::Detail::Approx(0).epsilon(tolerance));
    
    return w;
}


TEST_CASE("algorithm-translation")
{
    namespace ia = imagealign;
    
    cv::Mat target(100, 100, CV_8UC1);
    cv::randu(target, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::blur(target, target, cv::Size(5,5));
    cv::Mat tmpl = target(cv::Rect(20, 20, 10, 10));
    
    
    // Floating point
    {
        typedef ia::WarpTranslationF W;
        
        W::Traits::ParamType expected(20, 20);
        
        ia::WarpTranslationF w;
        w.setParameters(W::Traits::ParamType(18, 18));
        
        testAlgorithm< ia::AlignForwardAdditive<W> >(tmpl, target, w, 1, expected);
        testAlgorithm< ia::AlignForwardAdditive<W> >(tmpl, target, w, 2, expected);
        
        testAlgorithm< ia::AlignForwardCompositional<W> >(tmpl, target, w, 1, expected);
        testAlgorithm< ia::AlignForwardCompositional<W> >(tmpl, target, w, 2, expected);
        
        testAlgorithm< ia::AlignInverseCompositional<W> >(tmpl, target, w, 1, expected);
        testAlgorithm< ia::AlignInverseCompositional<W> >(tmpl, target, w, 2, expected);
    }
    
    // Double precision floating point
    {
        typedef ia::WarpTranslationD W;
        
        W::Traits::ParamType expected(20, 20);
        
        ia::WarpTranslationD w;
        w.setParameters(W::Traits::ParamType(18, 18));
        
        testAlgorithm< ia::AlignForwardAdditive<W> >(tmpl, target, w, 1, expected);
        testAlgorithm< ia::AlignForwardAdditive<W> >(tmpl, target, w, 2, expected);
        
        testAlgorithm< ia::AlignForwardCompositional<W> >(tmpl, target, w, 1, expected);
        testAlgorithm< ia::AlignForwardCompositional<W> >(tmpl, target, w, 2, expected);
        
        testAlgorithm< ia::AlignInverseCompositional<W> >(tmpl, target, w, 1, expected);
        testAlgorithm< ia::AlignInverseCompositional<W> >(tmpl, target, w, 2, expected);
    }
}

TEST_CASE("algorithm-euclidean")
{
    namespace ia = imagealign;
    
    cv::Mat target(100, 100, CV_8UC1);
    cv::randu(target, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::blur(target, target, cv::Size(5,5));
    
    cv::Mat tmpl;
    
    // Floating point
    {
        typedef ia::WarpEuclideanF W;
        
        W::Traits::ParamType expected(10.f, 15.f, 0.18f);
        W::Traits::ParamType noise(1.5f, -1.2f, 0.02f);
        
        W w;
        w.setParameters(expected);
        ia::warpImage<uchar, ia::SAMPLE_BILINEAR>(target, tmpl, cv::Size(20, 20), w);
        
        w.setParameters(w.parameters() + noise);
        
        testAlgorithm< ia::AlignForwardAdditive<W> >(tmpl, target, w, 1, expected);
        testAlgorithm< ia::AlignForwardAdditive<W> >(tmpl, target, w, 2, expected);
        
        testAlgorithm< ia::AlignForwardCompositional<W> >(tmpl, target, w, 1, expected);
        testAlgorithm< ia::AlignForwardCompositional<W> >(tmpl, target, w, 2, expected);
        
        testAlgorithm< ia::AlignInverseCompositional<W> >(tmpl, target, w, 1, expected);
        testAlgorithm< ia::AlignInverseCompositional<W> >(tmpl, target, w, 2, expected);
    }
    
    // Double precision floating point
    {
        typedef ia::WarpEuclideanD W;
        
        W::Traits::ParamType expected(10.f, 15.f, 0.18f);
        W::Traits::ParamType noise(1.5f, -1.2f, 0.02f);
        
        W w;
        w.setParameters(expected);
        ia::warpImage<uchar, ia::SAMPLE_BILINEAR>(target, tmpl, cv::Size(20, 20), w);
        
        w.setParameters(w.parameters() + noise);
        
        testAlgorithm< ia::AlignForwardAdditive<W> >(tmpl, target, w, 1, expected);
        testAlgorithm< ia::AlignForwardAdditive<W> >(tmpl, target, w, 2, expected);
        
        testAlgorithm< ia::AlignForwardCompositional<W> >(tmpl, target, w, 1, expected);
        testAlgorithm< ia::AlignForwardCompositional<W> >(tmpl, target, w, 2, expected);
        
        testAlgorithm< ia::AlignInverseCompositional<W> >(tmpl, target, w, 1, expected);
        testAlgorithm< ia::AlignInverseCompositional<W> >(tmpl, target, w, 2, expected);
    }
}

TEST_CASE("algorithm-similarity")
{
    namespace ia = imagealign;
    
    cv::Mat target(100, 100, CV_8UC1);
    cv::randu(target, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::blur(target, target, cv::Size(5,5));
    
    cv::Mat tmpl;
    
    // Floating point
    {
        typedef ia::WarpSimilarityF W;
        
        W::Traits::ParamType expectedCanonical(10.f, 15.f, 0.18f, 1.f);
        W::Traits::ParamType noiseCanonical(0.8f, -0.7f, 0.02f, 0.01f);
        
        W w;
        w.setParametersInCanonicalRepresentation(expectedCanonical);
        ia::warpImage<uchar, ia::SAMPLE_BILINEAR>(target, tmpl, cv::Size(20, 20), w);
        W::Traits::ParamType expected = w.parameters();
        
        w.setParametersInCanonicalRepresentation(w.parametersInCanonicalRepresentation() + noiseCanonical);
        
        testAlgorithm< ia::AlignForwardAdditive<W> >(tmpl, target, w, 1, expected, 0.02);
        testAlgorithm< ia::AlignForwardAdditive<W> >(tmpl, target, w, 2, expected, 0.02);
        
        testAlgorithm< ia::AlignForwardCompositional<W> >(tmpl, target, w, 1, expected, 0.02);
        testAlgorithm< ia::AlignForwardCompositional<W> >(tmpl, target, w, 2, expected, 0.02);
        
        testAlgorithm< ia::AlignInverseCompositional<W> >(tmpl, target, w, 1, expected, 0.02);
        testAlgorithm< ia::AlignInverseCompositional<W> >(tmpl, target, w, 2, expected, 0.02);
    }
    
    // Double precision floating point
    {
        typedef ia::WarpSimilarityD W;
        
        W::Traits::ParamType expectedCanonical(10.f, 15.f, 0.18f, 1.f);
        W::Traits::ParamType noiseCanonical(0.8f, -0.7f, 0.02f, 0.01f);
        
        W w;
        w.setParametersInCanonicalRepresentation(expectedCanonical);
        ia::warpImage<uchar, ia::SAMPLE_BILINEAR>(target, tmpl, cv::Size(20, 20), w);
        W::Traits::ParamType expected = w.parameters();
        
        w.setParametersInCanonicalRepresentation(w.parametersInCanonicalRepresentation() + noiseCanonical);
        
        testAlgorithm< ia::AlignForwardAdditive<W> >(tmpl, target, w, 1, expected, 0.02);
        testAlgorithm< ia::AlignForwardAdditive<W> >(tmpl, target, w, 2, expected, 0.02);
        
        testAlgorithm< ia::AlignForwardCompositional<W> >(tmpl, target, w, 1, expected, 0.02);
        testAlgorithm< ia::AlignForwardCompositional<W> >(tmpl, target, w, 2, expected, 0.02);
        
        testAlgorithm< ia::AlignInverseCompositional<W> >(tmpl, target, w, 1, expected, 0.02);
        testAlgorithm< ia::AlignInverseCompositional<W> >(tmpl, target, w, 2, expected, 0.02);
    }
}