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

#include <imagealign/imagealign.h>
#include <opencv2/opencv.hpp>

namespace ia = imagealign;

template<class Scalar>
cv::Point_<Scalar> toP(const cv::Matx<Scalar, 2, 1> &p) {
    return cv::Point_<Scalar>(p(0), p(1));
}

template<int WarpType, class Scalar>
void drawRectOfTemplate(cv::Mat &img, const ia::Warp<WarpType, Scalar> &w, cv::Size tplSize, cv::Scalar color)
{
    typedef typename ia::WarpTraits<WarpType, Scalar>::PointType PointType;

    PointType c0 = w(PointType(0, 0));
    PointType c1 = w(PointType(0 + tplSize.width, 0));
    PointType c2 = w(PointType(0 + tplSize.width, 0 + tplSize.height));
    PointType c3 = w(PointType(0, 0 + tplSize.height));

    cv::line(img, toP(c0), toP(c1), color, 1, CV_AA);
    cv::line(img, toP(c1), toP(c2), color, 1, CV_AA);
    cv::line(img, toP(c2), toP(c3), color, 1, CV_AA);
    cv::line(img, toP(c3), toP(c0), color, 1, CV_AA);
}

void runReg(int id, float x, float y) 
{
    typedef ia::WarpTranslationF W;
    typedef ia::AlignForwardCompositional<W> A;

    std::stringstream str1, str2;
    str1 << "reg" << id << "/template.png";
    str2 << "reg" << id << "/target.png";

    cv::Mat tpl = cv::imread(str1.str(), cv::IMREAD_GRAYSCALE);
    cv::Mat target = cv::imread(str2.str(), cv::IMREAD_GRAYSCALE);

    W w;
    W::Traits::ParamType p;
    p << x, y;
    w.setParameters(p);

    // cv::Mat wtarget;
    // ia::warpImage<uchar, ia::SAMPLE_BILINEAR>(target, wtarget, tpl.size(), w);
    // cv::imshow("wtpl", wtarget);
    
    std::vector<W> incrementals;

    A a;
    a.prepare(tpl, target, w, 3);
    a.align(w, 30, 0.003f, &incrementals);
}

/*
TEST_CASE("regression1")
{
    runReg(1, 267.f, 237.f);
}

TEST_CASE("regression2")
{
    runReg(2, 625.f, 155.f);
}

TEST_CASE("regression3")
{
    runReg(3, 580.f, 153.f);
}*/
