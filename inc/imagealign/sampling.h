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

#ifndef IMAGE_ALIGN_SAMPLING_H
#define IMAGE_ALIGN_SAMPLING_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace imagealign {
    
    template<int SampleMethod>
    class Sampler {
    public:
        template<class ChannelType>
        inline ChannelType sample(const cv::Mat &img, float x, float y) const;
        
        template<class ChannelType>
        inline ChannelType sample(const cv::Mat &img, const cv::Point2f &p) const;
    };
    
    /** Perform bilinear sampling. */
    const int SAMPLE_BILINEAR = 0;
    /** Perform nearest neighbor sampling. */
    const int SAMPLE_NEAREST = 1;
    
    
    
    
    
    /**
        Bilinear image interpolation for single channel images.
     
        This library assumes pixel origins at pixel centers, hence the half-pixel
        offset in the beginning.
     */
    template<>
    class Sampler<SAMPLE_BILINEAR> {
    public:
        
        /**
            Bilinear sampling at image coordinates.
         */
        template<class ChannelType>
        inline ChannelType sample(const cv::Mat &img, float x, float y) const
        {
            x -= 0.5f;
            y -= 0.5f;
            
            const int ix = static_cast<int>(std::floor(x));
            const int iy = static_cast<int>(std::floor(y));
            
            int x0 = cv::borderInterpolate(ix, img.cols, cv::BORDER_REFLECT_101);
            int x1 = cv::borderInterpolate(ix + 1, img.cols, cv::BORDER_REFLECT_101);
            int y0 = cv::borderInterpolate(iy, img.rows, cv::BORDER_REFLECT_101);
            int y1 = cv::borderInterpolate(iy + 1, img.rows, cv::BORDER_REFLECT_101);
            
            float a = x - (float)ix;
            float b = y - (float)iy;
            
            const ChannelType f0 = img.at<ChannelType>(y0, x0);
            const ChannelType f1 = img.at<ChannelType>(y0, x1);
            const ChannelType f2 = img.at<ChannelType>(y1, x0);
            const ChannelType f3 = img.at<ChannelType>(y1, x1);
            
            return cv::saturate_cast<ChannelType>((f0 * (float(1) - a) + f1 * a) * (float(1) - b) +
                                                  (f2 * (float(1) - a) + f3 * a) * b);
        }
        
        /**
            Bilinear sampling at image coordinates.
         */
        template<class ChannelType>
        inline ChannelType sample(const cv::Mat &img, const cv::Point2f &p) const
        {
            return sample<ChannelType>(img, p.x, p.y);
        }
    };
    
    /**
        Nearest neighbor image interpolation for single channel images.
     
        This library assumes pixel origins at pixel centers, hence the half-pixel
        offset in the beginning.
     */
    template<>
    class Sampler<SAMPLE_NEAREST> {
    public:
        
        /**
            Nearest sampling at image coordinates.
         */
        template<class ChannelType>
        inline ChannelType sample(const cv::Mat &img, float x, float y) const
        {
            x -= 0.5f;
            y -= 0.5f;
            
            const int ix = static_cast<int>(std::floor(x));
            const int iy = static_cast<int>(std::floor(y));
            
            int x0 = cv::borderInterpolate(ix, img.cols, cv::BORDER_REFLECT_101);
            int y0 = cv::borderInterpolate(iy, img.rows, cv::BORDER_REFLECT_101);
            
            return img.at<ChannelType>(y0, x0);
        }
        
        /**
            Nearest sampling at image coordinates.
         */
        template<class ChannelType>
        inline ChannelType sample(const cv::Mat &img, const cv::Point2f &p) const
        {
            return sample<ChannelType>(img, p.x, p.y);
        }
    };
}

#endif