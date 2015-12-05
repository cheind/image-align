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

#ifndef IMAGE_ALIGN_ALIGN_BASE_H
#define IMAGE_ALIGN_ALIGN_BASE_H

#include <imagealign/warp.h>
#include <opencv2/core/core.hpp>

namespace imagealign {
    
    /** 
        Base class for alignment algorithms.
     */
    template<class Derived, int WarpMode>
    class AlignBase {
    public:
        
        typedef AlignBase<Derived, WarpMode> SelfType;
        
        /** 
            Prepare for alignment.
         
            This function takes the template and target image and performs
            necessary pre-calculations to speed up the alignment process.
         
            \param tmpl Single channel template image
            \param target Single channel target image to align template with.
         */
        void prepare(cv::InputArray tmpl, cv::InputArray target)
        {
            // Do the basic thing everyone needs
            CV_Assert(tmpl.channels() == 1);
            CV_Assert(target.channels() == 1);
            
            tmpl.getMat().convertTo(_template, CV_32F);
            target.getMat().convertTo(_target, CV_32F);
            
            _error = std::numeric_limits<float>::max();
            _inc = WarpTraits<WarpMode>::ParamType::zeros();
            _iter = 0;
            
            // Invoke prepare of derived
            static_cast<Derived*>(this)->prepareImpl();
        }
        
        /** 
            Perform a single alignment step.
         
            This method takes the current state of the warp parameters and refines
            them by minimizing the energy function of the derived class.
         
            \param w Current state of warp estimation. Will be modified to hold result.
         */
        SelfType &align(Warp<WarpMode> &w)
        {
            static_cast< Derived*>(this)->alignImpl(w);
            ++_iter;
            
            return *this;
        }
        
        /**
            Perform multiple iterations of alignement until a stopping criterium is reached.
         
            This method takes the current state of the warp parameters and refines
            them by minimizing the energy function of the derived class.
         
            \param w Current state of warp estimation. Will be modified to hold result.
            \param maxIterations Stops after maxIterations have been performed.
            \param eps Stops when the norm of the incremental parameter update is below this value.
         */
        SelfType &align(Warp<WarpMode> &w, int maxIterations, float eps)
        {
            for (int i  = 0; i < maxIterations; ++i) {
                align(w);
                if (cv::norm(lastIncrement()) < eps)
                    break;
            }
            
            return *this;
        }
        
        /**
            Access the error value from last iteration.
         
            \return the error value corresponding to last invocation of align.
        */
        float lastError() const {
            return _error;
        }
        
        /**
            Access the number of iterations performed.
         
            The number of iterations is counted from last invocation of prepare.
        */
        int iteration() const {
            return _iter;
        }
        
        /** 
            Access the incremental warp parameter update from last iteration.
        */
        typename WarpTraits<WarpMode>::ParamType lastIncrement() const {
            return _inc;
        }
        
    protected:
        
        void setLastError(float err) {
            _error = err;
        }
        
        void setLastIncrement(typename WarpTraits<WarpMode>::ParamType &inc) {
            _inc = inc;
        }
        
        cv::Mat templateImage() {
            return _template;
        }
        
        cv::Mat targetImage() {
            return _target;
        }
        
    private:
        
        cv::Mat _template;
        cv::Mat _target;
        int _iter;
        float _error;
        typename WarpTraits<WarpMode>::ParamType _inc;
    };
    
    
}

#endif