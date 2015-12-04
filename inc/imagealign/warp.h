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

#ifndef IMAGE_ALIGN_WARP_H
#define IMAGE_ALIGN_WARP_H

#include <imagealign/bilinear.h>
#include <opencv2/core/core.hpp>
#include <iostream>

namespace imagealign {
    
    /** Set of supported parametrized warps.
        
        The warp type defines the motion that image coordinates can undergo. Simpler 
        motions take less parameters and may lead to more efficient algorithms, but
        depending on your application a simple motion model might not describe the true
        motion adequately and can therefore lead to less accurate results.
     
        A good overview of the supported motions is given in [1] section 2.1
     
        ## References
     
        [1] Szeliski, Richard.
            "Image alignment and stitching: A tutorial."
            Foundations and Trends® in Computer Graphics and Vision 2.1 (2006): 1-104.
     
    */
    enum EWarpType {
        WARP_TRANSLATION
    };
    
    
    /** Base class for planar motions */
    class PlanarWarp {
    public:
        
        typedef cv::Matx<float, 3, 3> MType;
        
        inline PlanarWarp() {
            setIdentity();
        }
        
        inline void setIdentity() {
            _m = MType::eye();
        }
        
        inline MType getMatrix() const {
            return _m;
        }
        
        inline void setMatrix(const MType &m) {
            _m = m;
        }
        
        /** Warp point */
        inline cv::Point2f operator()(const cv::Point2f &p) const {
            cv::Point3f x(p.x, p.y, 1.f);
            x = _m * x;
            return cv::Point2f(x.x / x.z, x.y / x.z);
        }

    protected:
        MType _m;
    };
    
    template<int WarpType>
    class Warp;
    
    /** Implementation of a warp supporting translational motion. */
    template<>
    class Warp<WARP_TRANSLATION> : public PlanarWarp {
    public:
        enum {
            NParameters = 2
        };
        
        /** Type to hold the warp parameters. */
        typedef cv::Matx<float, NParameters, 1> VType;
        
        /** Type to hold the Jacobian of the warp. */
        typedef cv::Matx<float, 2, NParameters> JType;
        
        
        /** Get warp parameters */
        VType getParameters() const {
            return VType(_m(0, 2), _m(1, 2));
        }
        
        /** Set warp parameters */
        void setParameters(const VType &p) {
            _m(0, 2) = p(0, 0);
            _m(1, 2) = p(1, 0);
        }
        
        /** Compute the jacobian of the warp.
            
            The Jacobian matrix contains the partial derivatives of the warp parameters
            with respect to x and y coordinates, evaluated at the current value of parameters.
            In this case:
                
                   tx   ty
                x   1    0
                y   0    1
         
         */
        JType jacobian() const {
            JType j = JType::zeros();
            j(0, 0) = 1.f;
            j(1, 1) = 1.f;
            return j;
        }
    };
    
    
    /**
        Warp image using bilinear interpolation.
     
        This method warps a given source image onto a given destination image. It assumes the direction
        of the warp is such that for given pixel in the destination image, the warp reports the corresponding
        pixel in the source image.
     
        This method will call create on the destination image.
     
        \param src_ Source image
        \param dst_ Destination image
        \param dstSize Size of destination image
        \param w Warp function
     */
    template<class ChannelType, int WarpType>
    void warpImage(cv::InputArray src_, cv::OutputArray dst_, cv::Size dstSize, const Warp<WarpType> &w)
    {
        CV_Assert(src_.channels() == 1);
        
        dst_.create(dstSize, src_.type());
        
        cv::Mat src = src_.getMat();
        cv::Mat dst = dst_.getMat();
        
        for (int y = 0; y < dstSize.height; ++y) {
            ChannelType *r = dst.ptr<ChannelType>(y);
            
            for (int x = 0; x < dstSize.width; ++x) {
                cv::Point2f wp = w(cv::Point2f(x + 0.5f, y + 0.5f));
                r[x] = bilinear<ChannelType>(src, wp);
            }
        }
    }
    
    
    
    
    
    
}

#endif