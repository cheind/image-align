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
    template<int WarpMode>
    class AlignInverseCompositional : public AlignBase<AlignInverseCompositional<WarpMode>, WarpMode> {
    protected:
        
        /** 
            Prepare for alignment.
         
            In the forward compositional algorithm only the Jacobian of the warp can be precomputed.
         */
        void prepareImpl()
        {
            typedef typename WarpTraits<WarpMode>::HessianType HessianType;
            typedef typename WarpTraits<WarpMode>::PixelSDIType PixelSDIType;
            typedef typename WarpTraits<WarpMode>::JacobianType JacobianType;
            
            Warp<WarpMode> w;
            w.setIdentity();
            
            _sdiPyramid.resize(this->numLevels());
            _invHessians.resize(this->numLevels());
            
            for (int i = 0; i < this->numLevels(); ++i) {
                float scale = this->scaleUpFactor();
                
                cv::Mat tpl = this->templateImagePyramid()[i];
                
                _sdiPyramid[i].resize(tpl.size().area());
                
                HessianType hessian = HessianType::zeros();
                
                int idx = 0;
                for (int y = 0; y < tpl.rows; ++y) {
                    for (int x = 0; x < tpl.cols; ++x, ++idx) {
                        cv::Point2f p(x + 0.5f, y + 0.5f);
                        
                        // 1. Compute the gradient of the template
                        const cv::Matx<float, 1, 2> grad = gradient<float, SAMPLE_NEAREST>(tpl, p);
                        
                        // 2. Evaluate the Jacobian of image location.
                        // Note: Jacobians are computed with pixel positions corresponding
                        // to the finest pyramid level.
                        JacobianType jacobian = w.jacobian(p * scale);
                        
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
        void alignImpl(Warp<WarpMode> &w)
        {
            cv::Mat tpl = this->templateImage();
            cv::Mat target = this->targetImage();
            
            typedef typename WarpTraits<WarpMode>::ParamType ParamType;
            
            ParamType b = ParamType::zeros();
            
            Sampler<SAMPLE_BILINEAR> s;

            float sumErrors = 0.f;
            int idx = 0;
            for (int y = 0; y < tpl.rows; ++y) {
                
                const float *tplRow = tpl.ptr<float>(y);
                
                for (int x = 0; x < tpl.cols; ++x, ++idx) {
                    cv::Point2f ptpl(x + 0.5f, y + 0.5f);
                    const float templateIntensity = tplRow[x];
                    
                    // 1. Warp target pixel back to template using w
                    cv::Point2f ptgt = this->scaleDown(w(this->scaleUp(ptpl)));
                    const float targetIntensity = s.sample<float>(target, ptgt);
                    
                    // 2. Compute the error. Roles reverse compared to forward additive / compositional
                    const float err = targetIntensity - templateIntensity;
                    sumErrors += err * err;
                    
                    // 3. Update b using SDI lookup
                    b += _sdiPyramid[this->level()][idx].t() * err;
                }
            }
            
            // 4. Solve Ax = b
            ParamType delta = _invHessians[this->level()] * b;
            
            // 5. Inverse compositional update of warp parameters.
            w.updateInverseCompositional(delta);
            
            this->setLastError(sumErrors / tpl.size().area());
            this->setLastIncrement(delta);
        }
        
    private:
        friend class AlignBase< AlignInverseCompositional<WarpMode>, WarpMode>;
        
        typedef std::vector< typename WarpTraits<WarpMode>::PixelSDIType > VecOfSDI;
        typedef std::vector< typename WarpTraits<WarpMode>::HessianType > VecOfHessian;
    
        std::vector<VecOfSDI> _sdiPyramid;
        VecOfHessian _invHessians;
        
    };
    
    
}

#endif