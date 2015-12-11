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

#ifndef IMAGE_ALIGN_INVERSE_COMPOSITIONAL_H
#define IMAGE_ALIGN_INVERSE_COMPOSITIONAL_H

#include <imagealign/align_base.h>
#include <imagealign/sampling.h>
#include <imagealign/gradient.h>
#include <opencv2/core/core.hpp>
#include <iostream>

namespace imagealign {
    
    /** 
        Inverse-compositional image alignment.
        
        'Best' aligns a template image with a target image through minimization of the sum of 
        squared intensity errors between the warped target image and the template image with 
        respect to the warp parameters.
     
        This algorithm is a variant of the classic Lucas-Kanade version. Baker an Matthews 
        coined it the inverse compositional, because of the way the incremental warp parameters
        are combined and the direction the warp is performed in. 
        Instead formulating the least squares equations by means of a parameter delta that is added to 
        the previous estimate:
     
            W(x, p) = W(x, p + delta)
     
        it is rewritten in terms of the template image and a warp composition
        
            W(x, p) = W(x, p) * W(x, delta)^-1 = W(W(x, delta)^-1, p)
     
        The inverse W(x, delta)^-1 stems from the fact that the delta is computed in the reverse direction,
        i.e in terms of the template image. The inverse then reverses this motion. Expressing the delta
        motion in terms of the template image allows us to transfer the following computations into pre-
        processing step
            - The Jacobian is evaluated at W(x, 0) for every template pixel.
            - The gradient is taken from the template image.
            - The pixel wise steepest descent images (SDI) are computed from the template image.
            - The Hessian is computed from the SDI above.
     
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
    class AlignInverseCompositional : public AlignBase< AlignInverseCompositional<W>, W > {
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
            
            _sdiPyramid.resize(this->numLevels());
            _invHessians.resize(this->numLevels());
            
            for (int i = 0; i < this->numLevels(); ++i) {
                ScalarType scale = this->scaleUpFactor(i);
                
                cv::Mat tpl = this->templateImagePyramid()[i];
                cv::Size s = tpl.size();
                
                _sdiPyramid[i].resize((s.width-2) * (s.height-2));
                
                HessianType hessian = W::Traits::zeroHessian(w.numParameters());
                
                int idx = 0;
                for (int y = 1; y < tpl.rows - 1 ; ++y) {
                    for (int x = 1; x < tpl.cols - 1; ++x, ++idx) {
                        PointType p;
                        p << ScalarType(x), ScalarType(y);
                        
                        // 1. Compute the gradient of the template
                        const GradientType grad = gradient<float, SAMPLE_NEAREST, typename W::Traits>(tpl, p);
                        
                        // 2. Evaluate the Jacobian of image location.
                        // Note: Jacobians are computed with pixel positions corresponding
                        // to the finest pyramid level.
                        JacobianType jacobian = w0.jacobian(p * scale);
                        
                        // 3. Compute steepest descent images
                        PixelSDIType sdi = grad * jacobian;
                        
                        // 4. Update inverse Hessian
                        hessian += sdi.t() * sdi;
                        
                        // 5. Store steepest descent images
                        _sdiPyramid[i][idx] = sdi;
                    }
                }

                // 6. Store inverse Hessian
                _invHessians[i] = hessian.inv();
                
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
            
            const ScalarType sUp = this->scaleUpFactor(this->level());
            const ScalarType sDown = ScalarType(1) / sUp;
            
            const VecOfSDI &sdi = _sdiPyramid[this->level()];
            
            Sampler<SAMPLE_BILINEAR> s;
            
            ParamType b = W::Traits::zeroParam(w.numParameters());
            ScalarType sumErrors = 0;
            int sumConstraints = 0;
            
            int idx = 0;
            for (int y = 1; y < tpl.rows - 1; ++y) {
                
                const float *tplRow = tpl.ptr<float>(y);
                
                for (int x = 1; x < tpl.cols - 1; ++x, ++idx) {
                    PointType ptpl;
                    ptpl << ScalarType(x), ScalarType(y);
                    const float templateIntensity = tplRow[x];
                    
                    // 1. Warp target pixel back to template using w
                    PointType ptgt = w(ptpl * sUp) * sDown;
                    
                    if (!this->isInImage(ptgt, target.size(), 1))
                        continue;
                    
                    const float targetIntensity = s.sample<float>(target, ptgt);
                    
                    // 2. Compute the error. Roles reverse compared to forward additive / compositional
                    const float err = targetIntensity - templateIntensity;
                    sumErrors += ScalarType(err * err);
                    sumConstraints += 1;
                    
                    // 3. Update b using SDI lookup
                    b += sdi[idx].t() * err;
                }
            }

            if (sumConstraints == 0) {
                this->setLastError(std::numeric_limits<ScalarType>::max());
                this->setLastIncrement(W::Traits::zeroParam(w.numParameters()));
                return;
            }
            
            // 4. Solve Ax = b
            ParamType delta = _invHessians[this->level()] * b;
            
            // 5. Inverse compositional update of warp parameters.
            w.updateInverseCompositional(delta);
            
            this->setLastError(sumErrors / sumConstraints);
            this->setLastIncrement(delta);
        }
        
    private:
        friend class AlignBase< AlignInverseCompositional<W>, W >;
        
        typedef std::vector< typename W::Traits::PixelSDIType > VecOfSDI;
        typedef std::vector< typename W::Traits::HessianType > VecOfHessian;
    
        std::vector<VecOfSDI> _sdiPyramid;
        VecOfHessian _invHessians;
        
    };
    
    
}

#endif