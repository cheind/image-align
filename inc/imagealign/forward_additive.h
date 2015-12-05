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

#ifndef IMAGE_ALIGN_FORWARD_ADDITIVE_H
#define IMAGE_ALIGN_FORWARD_ADDITIVE_H

#include <imagealign/align_base.h>
#include <imagealign/bilinear.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

namespace imagealign {
    
    /** 
        Forward-additive image alignment.
        
        'Best' aligns a template image with a target image through minimization of the sum of 
        squared intensity errors between the warped target image and the template image with 
        respect to the warp parameters.
     
        This algorithm is the classic algorithm proposed by Lucas-Kanade. Baker and Matthews 
        coined it later the forwards-additive algorithm because of its properties that the 
        direction of the warp is forward and warp parameters are summed.
     
        \tparam WarpType Type of warp motion to use during alignment. See EWarpType.
     
        ## Notes
     
        The implementation is not trimmed towards runtime nor memory efficiency. Some images
        are explicitely created for the sake of readability of the code.
     
        ## Based on
     
        [1] Lucas, Bruce D., and Takeo Kanade.
        "An iterative image registration technique with an application to stereo vision." 
        IJCAI. Vol. 81. 1981.
     
        [2] Baker, Simon, and Iain Matthews.
        "Lucas-kanade 20 years on: A unifying framework." 
        International journal of computer vision 56.3 (2004): 221-255.

     */
    template<int WarpMode>
    class AlignForwardAdditive : public AlignBase< AlignForwardAdditive<WarpMode>, WarpMode> {
    protected:
        
        /** 
            Prepare for alignment.
         
            In the forward additive algorithm not much data can be pre-calculated, which is
            why this algorithm is not the fastest. The only thing that could be calculated
            beforehand are the gradients of the target image.
         */
        void prepareImpl()
        {
            cv::Mat tpl = this->templateImage();
            cv::Mat target = this->targetImage();
            
            _warpedTarget.create(tpl.size(), CV_32FC1);
            _errorImage.create(tpl.size(), CV_32FC1);
            _warpedGradX.create(tpl.size(), CV_32FC1);
            _warpedGradY.create(tpl.size(), CV_32FC1);
            
            
            cv::Sobel(target, _gradX, CV_32F, 1, 0);
            cv::Sobel(target, _gradY, CV_32F, 0, 1);
            
            // Sobel uses 3x3 convolution matrix and result is not in units
            // of intensity anymore. Hence, normalize
            _gradX *= 0.125f;
            _gradY *= 0.125f;
        }
        
        /** 
            Perform a single alignment step.
         
            This method takes the current state of the warp parameters and refines
            them by minimizing the sum of squared intensity differences.
         
            \param w Current state of warp estimation. Will be modified to hold updated warp.
         */
        void alignImpl(Warp<WarpMode> &w)
        {
            cv::Mat tpl = this->templateImage();
            cv::Mat target = this->targetImage();
            
            // Warp target back to template with respect to current warp parameters
            warpImage<float>(target, _warpedTarget, _warpedTarget.size(), w);
            
            // Warp the gradient
            warpImage<float>(_gradX, _warpedGradX, _warpedGradX.size(), w);
            warpImage<float>(_gradY, _warpedGradY, _warpedGradY.size(), w);
            
            // Compute the error image
            _errorImage = tpl - _warpedTarget;
            
            typedef typename WarpTraits<WarpMode>::JacobianType JacobianType;
            typedef typename WarpTraits<WarpMode>::HessianType HessianType;
            typedef typename WarpTraits<WarpMode>::PixelSDIType PixelSDIType;
            typedef typename WarpTraits<WarpMode>::PixelSDITransposedType PixelSDITransposedType;
            
            JacobianType jacobian = w.jacobian();
            HessianType hessian = HessianType::zeros();
            PixelSDITransposedType sumSDITimesError = PixelSDITransposedType::zeros();
            
            // Loop over template region
            for (int y = 0; y < tpl.rows; ++y) {
                
                const float *gxRow = _warpedGradX.ptr<float>(y);
                const float *gyRow = _warpedGradY.ptr<float>(y);
                const float *eRow = _errorImage.ptr<float>(y);
                
                for (int x = 0; x < tpl.cols; ++x) {
                    const PixelSDIType sd = cv::Matx<float, 1, 2>(gxRow[x], gyRow[x]) * jacobian;
                    sumSDITimesError += (sd.t() * eRow[x]);
                    hessian += sd.t() * sd;
                }
            }
            
            typename WarpTraits<WarpMode>::ParamType delta = hessian.inv() * sumSDITimesError;
            
            // Additive warp parameter update.
            w.setParameters(w.getParameters() + delta);
            
            this->setLastError(cv::mean(_errorImage)[0]);
            this->setLastIncrement(delta);
        }
        
    private:
        friend class AlignBase< AlignForwardAdditive<WarpMode>, WarpMode>;
        
        cv::Mat _warpedTarget;
        cv::Mat _errorImage;
        cv::Mat _gradX, _gradY;
        cv::Mat _warpedGradX, _warpedGradY;
    };
    
    
}

#endif