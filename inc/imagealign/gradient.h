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

#ifndef IMAGE_ALIGN_GRADIENT_H
#define IMAGE_ALIGN_GRADIENT_H

#include <imagealign/bilinear.h>
#include <opencv2/core/core.hpp>

namespace imagealign {

    /** 
        Image gradient approximation.
     
        Approximates the image derivate in x and y direction for the given image coordinates.
        Approximation is based on central difference.
     */
    template<class ChannelType>
    inline cv::Matx<ChannelType, 1, 2> gradient(const cv::Mat &img, const cv::Point2f &p)
    {
        return cv::Matx<ChannelType, 1, 2>(
            (bilinear<ChannelType>(img, p.x + 1.f, p.y) - bilinear<ChannelType>(img, p.x - 1.f, p.y)) * 0.5f,
            (bilinear<ChannelType>(img, p.x, p.y + 1.f) - bilinear<ChannelType>(img, p.x, p.y - 1.f)) * 0.5f
        );
    }
    
}

#endif