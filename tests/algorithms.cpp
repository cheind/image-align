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
    
    a.align(w, 100, S(0));
    
    
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
        
        W w;
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
        
        W w;
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

// Test dummy dynamic warp;

namespace ia = imagealign;

namespace imagealign {
	const int WARP_TRANSLATION_DYAMIC = 255;

	template<class Scalar>
	struct WarpTraits<WARP_TRANSLATION_DYAMIC, Scalar> : WarpTraitsForRunTimeKnownParameterCount<WARP_TRANSLATION_DYAMIC, Scalar> {};

	template<class Scalar>
	class Warp<WARP_TRANSLATION_DYAMIC, Scalar> {
	public:
		typedef WarpTraits<WARP_TRANSLATION_DYAMIC, Scalar> Traits;

		Warp() {
			_m.create(2, 1);
			setIdentity();
		}

		Warp(const Warp<WARP_TRANSLATION_DYAMIC, Scalar> &other) {
			_m = other._m.clone();
		}

		int numParameters() const {
			return 2;
		}

		void setIdentity() {
			_m.setTo(0);
		}

		Warp<WARP_TRANSLATION_DYAMIC, Scalar> scaled(int numLevels) const
		{
			Scalar s = std::pow(Scalar(2), numLevels);
			Warp<WARP_TRANSLATION_DYAMIC, Scalar> ws(*this);
			ws._m *= s;
			return ws;
		}

		typename Traits::PointType operator()(const typename Traits::PointType &p) const {
			return typename Traits::PointType(p(0) + _m(0, 0), p(1) + _m(1, 0));
		}

		typename Traits::JacobianType jacobian(const typename Traits::PointType &p) const {
			return Traits::JacobianType::eye(2, 2, CV_MAKETYPE(cv::DataType<Scalar>::depth, 1));
		}

		void updateInverseCompositional(const typename Traits::ParamType &delta) {
			_m -= delta;
		}

		void updateForwardAdditive(const typename Traits::ParamType &delta) {
			_m += delta;
		}

		void updateForwardCompositional(const typename Traits::ParamType &delta) {
			_m += delta;
		}

		// Helper functions

		void setParameters(const typename Traits::ParamType &p) {
			p.copyTo(_m);
		}

		typename Traits::ParamType parameters() const {
			return _m.clone();
		}

	private:
		cv::Mat_<Scalar> _m;
	};
}

TEST_CASE("algorithm-dynamic-warp")
{
    namespace ia = imagealign;
    
    cv::Mat target(100, 100, CV_8UC1);
    cv::randu(target, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::blur(target, target, cv::Size(5,5));
    cv::Mat tmpl = target(cv::Rect(20, 20, 10, 10));
    
    
    // Floating point
    {
        typedef ia::Warp<ia::WARP_TRANSLATION_DYAMIC, float> W;
        
        W::Traits::ParamType expected(2, 1, CV_32FC1);
        expected.at<float>(0, 0) = 20;
        expected.at<float>(1, 0) = 20;
        
        W::Traits::ParamType noisy(2, 1, CV_32FC1);
        noisy.at<float>(0, 0) = 19;
        noisy.at<float>(1, 0) = 19;
        
        W w;
        w.setParameters(noisy);
        
        testAlgorithm< ia::AlignForwardAdditive<W> >(tmpl, target, w, 1, expected);
        testAlgorithm< ia::AlignForwardAdditive<W> >(tmpl, target, w, 2, expected);
        
        testAlgorithm< ia::AlignInverseCompositional<W> >(tmpl, target, w, 1, expected);
        testAlgorithm< ia::AlignInverseCompositional<W> >(tmpl, target, w, 2, expected);
        
        testAlgorithm< ia::AlignForwardCompositional<W> >(tmpl, target, w, 1, expected);
        testAlgorithm< ia::AlignForwardCompositional<W> >(tmpl, target, w, 2, expected);
    }
    
    // Double precision floating point
    {
        typedef ia::Warp<ia::WARP_TRANSLATION_DYAMIC, double> W;
        
        W::Traits::ParamType expected(2, 1, CV_64FC1);
        expected.at<double>(0, 0) = 20;
        expected.at<double>(1, 0) = 20;
        
        W::Traits::ParamType noisy(2, 1, CV_64FC1);
        noisy.at<double>(0, 0) = 19;
        noisy.at<double>(1, 0) = 19;
        
        W w;
        w.setParameters(noisy);
        
        testAlgorithm< ia::AlignForwardAdditive<W> >(tmpl, target, w, 1, expected);
        testAlgorithm< ia::AlignForwardAdditive<W> >(tmpl, target, w, 2, expected);
        
        testAlgorithm< ia::AlignInverseCompositional<W> >(tmpl, target, w, 1, expected);
        testAlgorithm< ia::AlignInverseCompositional<W> >(tmpl, target, w, 2, expected);
        
        testAlgorithm< ia::AlignForwardCompositional<W> >(tmpl, target, w, 1, expected);
        testAlgorithm< ia::AlignForwardCompositional<W> >(tmpl, target, w, 2, expected);
    }

}
