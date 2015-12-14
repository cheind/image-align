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
        
        typedef typename W::Traits::ParamType ParamType;
        typedef typename W::Traits::HessianType HessianType;
        typedef typename W::Traits::PixelSDIType PixelSDIType;
        typedef typename W::Traits::GradientType GradientType;
        typedef typename W::Traits::JacobianType JacobianType;
        typedef typename W::Traits::PointType PointType;
        typedef typename W::Traits::ScalarType ScalarType;
        
        /** 
            Prepare for alignment.
         
            In the forward compositional algorithm only the Jacobian of the warp can be precomputed.
         */
        void prepareImpl(const W &w)
        {
            W w0(w);
            w0.setIdentity();
            
            _jacobianPyramid.resize(this->numLevels());
            
            for (int i = 0; i < this->numLevels(); ++i) {

                cv::Size s = this->templateImagePyramid()[i].size();
            
                _jacobianPyramid[i].resize((s.width-2) * (s.height-2));
            
                int idx = 0;
                for (int y = 1; y < s.height - 1; ++y) {
                    for (int x = 1; x < s.width - 1; ++x, ++idx) {
                        _jacobianPyramid[i][idx] = w0.jacobian(PointType(ScalarType(x), ScalarType(y)));
                    }
                }

                w0 = w0.scaled(-1);
            }
        }
        
        /** 
            Perform a single alignment step.
         
            This method takes the current state of the warp parameters and refines
            them by minimizing the sum of squared intensity differences.
         
            \param w Current state of warp estimation. Will be modified to hold updated warp.
         */
        SingleStepResult<W> alignImpl(W &w)
        {
            cv::Mat tpl = this->templateImage();
            cv::Mat target = this->targetImage();
            
            // Computing the gradient happens on the warped image. Since evaluating the
            // the gradient in both directions takes 4 bilinear lookups, we are better off
            // warping the entire target image explicitely here.
            warpImage<float, SAMPLE_BILINEAR>(target, _warpedTargetImage, tpl.size(), w);
            
            HessianType hessian = W::Traits::zeroHessian(w.numParameters());
            ParamType b = W::Traits::zeroParam(w.numParameters());
            
            Sampler<SAMPLE_NEAREST> s;

            ScalarType sumErrors = 0;
            int sumConstraints = 0;
            
            int idx = 0;
            for (int y = 1; y < tpl.rows - 1; ++y) {
                
                const float *tplRow = tpl.ptr<float>(y);
                
                for (int x = 1; x < tpl.cols - 1; ++x, ++idx) {
                    PointType ptpl;
                    ptpl << ScalarType(x), ScalarType(y);
                    const float templateIntensity = tplRow[x];
                    
                    // 1. Lookup the target intensity using the already back warped image.
                    const float targetIntensity = s.sample<float>(_warpedTargetImage, ptpl);
                    
                    // 2. Compute the error
                    const float err = templateIntensity - targetIntensity;
                    sumErrors += ScalarType(err * err);
                    sumConstraints += 1;
                    
                    // 3. Compute the target gradient on the warped image
                    const GradientType grad = gradient<float, SAMPLE_NEAREST, typename W::Traits>(_warpedTargetImage, ptpl);
                    
                    // 4. Lookup the prec-computed Jacobian for the template pixel position corresponding to finest level.
                    const JacobianType &jacobian = _jacobianPyramid[this->level()][idx];
                    
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
            
            SingleStepResult<W> step;
            step.delta = delta;
            step.sumErrors = sumErrors;
            step.numConstraints = sumConstraints;
            
            return step;
        }
        
        void applyStep(W &w, const SingleStepResult<W> &s) {
            w.updateForwardCompositional(s.delta);
        }
        
    private:
        friend class AlignBase< AlignForwardCompositional<W>, W>;
        
        typedef std::vector< typename W::Traits::JacobianType > VecOfJacobians;
        std::vector<VecOfJacobians> _jacobianPyramid;
        
        cv::Mat _warpedTargetImage;
    };
    
    
}

#endif