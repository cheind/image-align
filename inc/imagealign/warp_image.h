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

#ifndef IMAGE_ALIGN_WARP_IMAGE_H
#define IMAGE_ALIGN_WARP_IMAGE_H

#include <imagealign/sampling.h>
#include <imagealign/warp.h>
#include <opencv2/core/core.hpp>

namespace imagealign {

    /**
        Warp an image using bilinear interpolation.
     
        This method warps a given source image onto a given destination image. It assumes the direction
        of the warp is such that for given pixel in the destination image, the warp reports the corresponding
        pixel in the source image.
     
        This method will call create on the destination image.
     
        \param src_ Source image
        \param dst_ Destination image
        \param dstSize Size of destination image
        \param s Sampler to use.
        \param w Warp function
     */
    template<class ChannelType, int SampleMethod, int WarpType, class Scalar>
    void warpImage(cv::InputArray src_, cv::OutputArray dst_, cv::Size dstSize, const Warp<WarpType, Scalar> &w, const Sampler<SampleMethod> &s = Sampler<SampleMethod>())
    {
        CV_Assert(src_.channels() == 1);
        
        typedef typename Warp<WarpType, Scalar>::Traits::PointType PointType;
        
        dst_.create(dstSize, src_.type());
        
        cv::Mat src = src_.getMat();
        cv::Mat dst = dst_.getMat();
        
        for (int y = 0; y < dstSize.height; ++y) {
            ChannelType *r = dst.ptr<ChannelType>(y);
            
            for (int x = 0; x < dstSize.width; ++x) {
                PointType wp = w(PointType(Scalar(x), Scalar(y)));
                r[x] = s.template sample<ChannelType>(src, wp);
            }
        }
    }
    
    
    
    
    
}

#endif