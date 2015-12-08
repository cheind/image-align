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

#ifndef IMAGE_IMAGE_PYRAMID_H
#define IMAGE_IMAGE_PYRAMID_H

#include <imagealign/config.h>
#include <vector>

IA_DISABLE_PRAGMA_WARN(4190)
IA_DISABLE_PRAGMA_WARN(4244)
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
IA_DISABLE_PRAGMA_WARN_END
IA_DISABLE_PRAGMA_WARN_END

namespace imagealign {
    
    /** 
        Hierarchical image pyramid.
     
        Lower levels correspond to coarser images. Levels are generated recursively,
        by smoothing and shrinking parent levels successively.
    */
    class ImagePyramid {
    public:
        
        /** Create image pyramid from image. */
        void create(cv::InputArray img, int levels) {
            
            levels = std::max<int>(levels, 1);
            _pyr.resize(levels);
            
            // All images are floating point
            img.getMat().convertTo(_pyr[0], CV_32F);
            
            for (int i = 1; i < levels; ++i) {
                cv::pyrDown(_pyr[i-1], _pyr[i]);
            }
            
            // By convention: coarser levels are lower numbers.
            std::reverse(_pyr.begin(), _pyr.end());
        }
        
        /**
            Access the number of levels in the pyramid
         */
        int numLevels() const {
            return (int)_pyr.size();
        }
        
        /** 
            Return the image corresponding to the i-th level.
         */
        cv::Mat operator[](size_t level) const {
            return _pyr[level];
        }
        
        /** 
            Return the maximum number of levels for image size.
        */
        static int maxLevelsForImageSize(cv::Size s) {
            int level = 0;
            while (s.width >= 10 && s.height >= 10) {
                s.width /= 2;
                s.height /= 2;
                ++level;
            }
            return level;
        }
        
    private:
        std::vector<cv::Mat> _pyr;
    };
    
}

#endif