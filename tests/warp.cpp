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

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <imagealign/warp.h>

TEST_CASE("warp-translational")
{
    namespace ia = imagealign;
    
    typedef ia::WarpTranslationF W;
    
    W w;
    w.setIdentity();
    
    REQUIRE(w.parameters()(0,0) == 0.f);
    REQUIRE(w.parameters()(1,0) == 0.f);
    
    W::Traits::ParamType p;
    p(0,0) = 10.f;
    p(1,0) = 5.f;
    w.setParameters(p);
    
    W::Traits::PointType x(5.f, 5.f);
    W::Traits::PointType wx = w(x);
    
    REQUIRE(wx(0) == 15.f);
    REQUIRE(wx(1) == 10.f);
    
    
    cv::Matx<float, 2, 2> j;
    j << 1, 0, 0, 1;
    REQUIRE(cv::norm(w.jacobian(W::Traits::PointType(10,10)) - j) == Catch::Detail::Approx(0));
}

TEST_CASE("warp-euclidean")
{
    namespace ia = imagealign;
    
    typedef ia::WarpEuclideanF W;
    
    W w;
    w.setIdentity();
    
    REQUIRE(w.parameters()(0,0) == 0.f);
    REQUIRE(w.parameters()(1,0) == 0.f);
    REQUIRE(w.parameters()(2,0) == 0.f);
    
    W::Traits::ParamType p;
    p(0,0) = 5.f;
    p(1,0) = 5.f;
    p(2,0) = 3.1415f;
    w.setParameters(p);
    
    W::Traits::PointType x(0.f, 0.f);
    W::Traits::PointType wx = w(x);
    
    REQUIRE(wx(0) == 5.f);
    REQUIRE(wx(1) == 5.f);
    
    
    x = W::Traits::PointType(10.f, 15.f);
    wx = w(x);
    
    REQUIRE(wx(0) == Catch::Detail::Approx(-10.f + 5.f).epsilon(0.01));
    REQUIRE(wx(1) == Catch::Detail::Approx(-15.f + 5.f).epsilon(0.01));
}

TEST_CASE("warp-similarity")
{
    namespace ia = imagealign;
    
    typedef ia::WarpSimilarityF W;
    
    W w;
    w.setIdentity();
    
    REQUIRE(w.parameters()(0,0) == 0.f);
    REQUIRE(w.parameters()(1,0) == 0.f);
    REQUIRE(w.parameters()(2,0) == 0.f);
    REQUIRE(w.parameters()(3,0) == 0.f);
    
    w.setParametersInCanonicalRepresentation(W::Traits::ParamType(5.f, 5.f, 1.7f, 2.0f));
    W::Traits::ParamType pr = w.parametersInCanonicalRepresentation();
    REQUIRE(pr(0,0) == Catch::Detail::Approx(5));
    REQUIRE(pr(1,0) == Catch::Detail::Approx(5));
    REQUIRE(pr(2,0) == Catch::Detail::Approx(1.7));
    REQUIRE(pr(3,0) == Catch::Detail::Approx(2));
    
    w.setParametersInCanonicalRepresentation(W::Traits::ParamType(5.f, 5.f, 3.1415f, 2.f));
    
    W::Traits::PointType x(0.f, 0.f);
    W::Traits::PointType wx = w(x);
    
    REQUIRE(wx(0) == 5.f);
    REQUIRE(wx(1) == 5.f);
    
    x = W::Traits::PointType(10.f, 15.f);
    wx = w(x);
    
    REQUIRE(wx(0) == Catch::Detail::Approx(-20.f + 5.f).epsilon(0.01));
    REQUIRE(wx(1) == Catch::Detail::Approx(-30.f + 5.f).epsilon(0.01));
}