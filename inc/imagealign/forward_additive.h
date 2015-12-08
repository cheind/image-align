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
#include <imagealign/warp.h>
#include <imagealign/sampling.h>
#include <imagealign/gradient.h>
#include <opencv2/core/core.hpp>

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
     
        ## Based on
     
        [1] Lucas, Bruce D., and Takeo Kanade.
        "An iterative image registration technique with an application to stereo vision." 
        IJCAI. Vol. 81. 1981.
     
        [2] Baker, Simon, and Iain Matthews.
        "Lucas-kanade 20 years on: A unifying framework." 
        International journal of computer vision 56.3 (2004): 221-255.

     */
    template<class W>
    class AlignForwardAdditive : public AlignBase< AlignForwardAdditive<W>, W> {
    protected:
        
        /** 
            Prepare for alignment.
         
            In the forward additive algorithm not much data can be pre-calculated, which is
            why this algorithm is not the fastest. The only thing that could be calculated
            beforehand are the gradients of the target image.
         */
        void prepareImpl()
        {
            // Nothing todo here. Gradient is computed on the fly.
        }
        
        /** 
            Perform a single alignment step.
         
            This method takes the current state of the warp parameters and refines
            them by minimizing the sum of squared intensity differences.
         
            \param w Current state of warp estimation. Will be modified to hold updated warp.
         */
        void alignImpl(W &w)
        {
            cv::Mat tpl = this->templateImage();
            cv::Mat target = this->targetImage();
            
            typedef typename W::Traits::JacobianType JacobianType;
            typedef typename W::Traits::HessianType HessianType;
            typedef typename W::Traits::PixelSDIType PixelSDIType;
            typedef typename W::Traits::ParamType ParamType;
            
            Sampler<SAMPLE_BILINEAR> s;
            
            HessianType hessian = HessianType::zeros();
            ParamType b = ParamType::zeros();

            float sumErrors = 0.f;
            
            for (int y = 0; y < tpl.rows; ++y) {
                
                const float *tplRow = tpl.ptr<float>(y);
                
                for (int x = 0; x < tpl.cols; ++x) {
                    cv::Point2f ptpl(x + 0.5f, y + 0.5f);
                    const float templateIntensity = tplRow[x];
                    
                    cv::Point2f ptplOrig = this->scaleUp(ptpl);
                    
                    // 1. Warp target pixel back to template using w
                    cv::Point2f ptgt = this->scaleDown(w(ptplOrig));
                    const float targetIntensity = s.sample<float>(target, ptgt);
                    
                    // 2. Compute the error
                    const float err = templateIntensity - targetIntensity;
                    sumErrors += err * err;
                    
                    // 3. Compute the target gradient warped back
                    const cv::Matx<float, 1, 2> grad = gradient<float, SAMPLE_BILINEAR>(target, ptgt);
                    
                    // 4. Compute the jacobian for the template pixel position
                    JacobianType jacobian = w.jacobian(ptplOrig);
                    
                    // 5. Compute the steepest descent image (SDI) for current pixel location
                    const PixelSDIType sd = grad * jacobian;
                    
                    // 6. Update running sum of SDI times error
                    b += sd.t() * err;
                    
                    // 7. Update Hessian
                    hessian += sd.t() * sd;
                }
            }
            
            // 8. Solve Ax = b
            ParamType delta = hessian.inv() * b;
            
            // 9. Additive update of warp parameters.
            w.updateForwardAdditive(delta);
            
            this->setLastError(sumErrors / tpl.size().area());
            this->setLastIncrement(delta);
        }
        
    private:
        friend class AlignBase< AlignForwardAdditive<W>, W>;
    };
    
    
}

#endif