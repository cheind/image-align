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

#ifndef IMAGE_ALIGN_FORWARD_COMPOSITIONAL_H
#define IMAGE_ALIGN_FORWARD_COMPOSITIONAL_H

#include <imagealign/align_base.h>
#include <imagealign/sampling.h>
#include <imagealign/gradient.h>
#include <imagealign/warp_image.h>
#include <opencv2/core/core.hpp>

namespace imagealign {
    
    /** 
        Forward-compositional image alignment.
        
        'Best' aligns a template image with a target image through minimization of the sum of 
        squared intensity errors between the warped target image and the template image with 
        respect to the warp parameters.
     
        This algorithm is a variant of the classic Lucas-Kanade version. Baker an Matthews 
        coined it the forward compositional, because of the way the incremental warp parameters
        are combined. Instead formulating the least squares equations by means of a parameter 
        delta that is added to the previous estimate:
     
            W(x, p) = W(x, p + delta)
     
        it is rewritten as the combination of two warps
        
            W(x, p) = W(x, p) * W(x, delta) = W(W(x, delta), p)
     
        where * is meant to be the composing operator (when dealing with planar motions, often
        a 3x3 matrix multiplication).
     
        This leads to three differences compared to the classic Lucas-Kanade algorithm
            - The Jacobian is evaluated at W(x, 0) and is thus constant that can be pre-computed for
              every pixel in the template.
            - The gradient is evaluated at the warped image. This is a somewhat subtle difference:
              in the classic version the gradient was evaluated on the original image and then
              warped.
            - The way the new warp is calculated is by composition rather than addition of parameters.
     
        \tparam WarpType Type of warp motion to use during alignment. See EWarpType.
     
        ## Based on
     
        [1] Baker, Simon, and Iain Matthews. 
            "Equivalence and efficiency of image alignment algorithms." 
            Computer Vision and Pattern Recognition, 2001. CVPR 2001.
     
        [2] Baker, Simon, and Iain Matthews. 
            Lucas-Kanade 20 years on: A unifying framework: Part 1.
            Technical Report CMU-RI-TR-02-16, Carnegie Mellon University Robotics Institute, 2002.

     */
    template<class W>
    class AlignForwardCompositional : public AlignBase< AlignForwardCompositional<W>, W> {
    protected:
        
        /** 
            Prepare for alignment.
         
            In the forward compositional algorithm only the Jacobian of the warp can be precomputed.
         */
        void prepareImpl()
        {
            W w;
            w.setIdentity();
            
            // Computing jacobians only for finest pyramid level.
            cv::Size s = this->templateImagePyramid().back().size();
            _jacobians.resize(s.area());
            
            int idx = 0;
            for (int y = 0; y < s.height; ++y) {
                for (int x = 0; x < s.width; ++x, ++idx) {
                    _jacobians[idx] = w.jacobian(cv::Point2f(x + 0.5f, y + 0.5f));
                }
            }
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
            
            int pixelScaleUp = (int)this->scaleUpFactor();
            
            typedef typename W::Traits::HessianType HessianType;
            typedef typename W::Traits::PixelSDIType PixelSDIType;
            typedef typename W::Traits::ParamType ParamType;
            typedef typename W::Traits::JacobianType JacobianType;
            
            // Computing the gradient happens on the warped image. Since evaluating the
            // the gradient in both directions takes 4 bilinear lookups, we are better off
            // warping the entire target image explicitely here.
            warpImage<float>(target, _warpedTargetImage, tpl.size(), w, this->scaleUpFactor(), this->scaleDownFactor());
            
            HessianType hessian = HessianType::zeros();
            ParamType b = ParamType::zeros();
            
            Sampler<SAMPLE_NEAREST> s;

            float sumErrors = 0.f;
            
            for (int y = 0; y < tpl.rows; ++y) {
                
                const float *tplRow = tpl.ptr<float>(y);
                
                const int idxRowOrig = (y * pixelScaleUp) * (tpl.cols * pixelScaleUp);
                
                for (int x = 0; x < tpl.cols; ++x) {
                    cv::Point2f ptpl(x + 0.5f, y + 0.5f);
                    const float templateIntensity = tplRow[x];
                    
                    // 1. Lookup the target intensity using the already back warped image.
                    const float targetIntensity = s.sample(_warpedTargetImage, ptpl);
                    
                    // 2. Compute the error
                    const float err = templateIntensity - targetIntensity;
                    sumErrors += err * err;
                    
                    // 3. Compute the target gradient on the warped image
                    const cv::Matx<float, 1, 2> grad = gradient<float, SAMPLE_NEAREST>(_warpedTargetImage, ptpl);
                    
                    // 4. Lookup the prec-computed Jacobian for the template pixel position corresponding to finest level.
                    const JacobianType &jacobian = _jacobians[idxRowOrig + x * pixelScaleUp];
                    
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
            
            // 9. Compositional update of warp parameters.
            w.updateForwardCompositional(delta);
            
            this->setLastError(sumErrors / tpl.size().area());
            this->setLastIncrement(delta);
        }
        
    private:
        friend class AlignBase< AlignForwardCompositional<W>, W>;
        
        typedef std::vector< typename W::Traits::JacobianType > VecOfJacobians;
        VecOfJacobians _jacobians;
        cv::Mat _warpedTargetImage;
    };
    
    
}

#endif