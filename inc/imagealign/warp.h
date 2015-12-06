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

#include <imagealign/sampling.h>
#include <opencv2/core/core.hpp>

namespace imagealign {
    
    /** 
        Set of supported warp functions.
        
        The warp mode defines the motion that image coordinates can undergo. Simpler
        motions take less parameters and may lead to more efficient algorithms, but
        depending on your application a simple motion model might not describe the true
        motion adequately and can therefore lead to less accurate results.
     
        A good overview of the supported motions is given in [1] section 2.1
     
        ## References
     
        [1] Szeliski, Richard.
            "Image alignment and stitching: A tutorial."
            Foundations and Trends in Computer Graphics and Vision 2.1 (2006): 1-104.
     
    */
    
    /** 2D translational motion. See Warp<WARP_TRANSLATION>. */
    const int WARP_TRANSLATION = 0;
    
    /** 2D Euclidean motion. See Warp<WARP_EUCLIDEAN>. */
    const int WARP_EUCLIDEAN = 1;
    
    /** 2D Similarity motion. See Warp<WARP_SIMILARITY>. */
    const int WARP_SIMILARITY = 2;
    
    
    /** 
        Base class for warps based on planar motions. 
     
        Planar motions in 2D can be well described by a 3x3 matrix, which is
        what this class does. It can thus provide a (not always) efficient warp
        method for image coordinates.
    */
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
    
    /** 
        Interface declaration for warps.
    */
    template<int WarpMode>
    class Warp;
    
    template<int WarpMode>
    struct WarpTraits;
    
    /** 
        Default warp traits implementation for compile time known parameter sizes.
     */
    template<int NParams>
    struct WarpTraitsForDoF {
        enum {
            NParameters = NParams
        };
        
        /** Type to hold parameters of warp. */
        typedef cv::Matx<float, NParameters, 1> ParamType;
        
        /** Type to hold Jacobian of warp. */
        typedef cv::Matx<float, 2, NParameters> JacobianType;
        
        /** Type to hold Hessian matrix. */
        typedef cv::Matx<float, NParameters, NParameters> HessianType;
        
        /** Type to hold the steepest descent image for a single pixel */
        typedef cv::Matx<float, 1, NParameters> PixelSDIType;

    };
    
    /**
        Warp traits for translational motion.
     */
    template<>
    struct WarpTraits<WARP_TRANSLATION> : WarpTraitsForDoF<2> {};
    
    /**
     Warp traits for Euclidean motion.
     */
    template<>
    struct WarpTraits<WARP_EUCLIDEAN> : WarpTraitsForDoF<3> {};
    
    /**
     Warp traits for Similarity motion.
     */
    template<>
    struct WarpTraits<WARP_SIMILARITY> : WarpTraitsForDoF<4> {};

    
    /** 
        Warp implementation for pure translational motion.
     
        A translational transform preserves orientation, lengths, angles, parallel lines 
        and straight lines.
     
        The warp is parametrized with 2 parameters (tx, ty). In matrix notation
     
            1  0  tx
            0  1  ty

    */
    template<>
    class Warp<WARP_TRANSLATION> : public PlanarWarp {
    public:
        typedef typename WarpTraits<WARP_TRANSLATION>::ParamType ParamType;
        typedef typename WarpTraits<WARP_TRANSLATION>::JacobianType JacobianType;

        /** Get warp parameters */
        ParamType getParameters() const {
            return ParamType(_m(0, 2), _m(1, 2));
        }
        
        /** Set warp parameters */
        void setParameters(const ParamType &p) {
            _m(0, 2) = p(0, 0);
            _m(1, 2) = p(1, 0);
        }
        
        /** Compute the jacobian of the warp.
            
            The Jacobian matrix contains the partial derivatives of the warp parameters
            with respect to x and y coordinates, evaluated at the given pixel position.
            In this case:
                
                   tx   ty
                x   1    0
                y   0    1
         
         */
        JacobianType jacobian(const cv::Point2f &p) const {
            JacobianType j = JacobianType::zeros();
            j(0, 0) = 1.f;
            j(1, 1) = 1.f;
            return j;
        }
    };
    
    /**
        Warp implementation for Euclidean motion.
     
        An Euclidean transform consists of  rotation and translation. It preserves lengths, 
        angles, parallel lines and straight lines.
     
        The warp is parametrized with 3 parameters (tx, ty, theta). In matrix notation
     
            c  -s  tx
            s   c  ty
     
            c = cos(theta)
            s = sin(theta)
     */
    template<>
    class Warp<WARP_EUCLIDEAN> : public PlanarWarp {
    public:
        
        typedef typename WarpTraits<WARP_EUCLIDEAN>::ParamType ParamType;
        typedef typename WarpTraits<WARP_EUCLIDEAN>::JacobianType JacobianType;
        
        /** Get warp parameters */
        ParamType getParameters() const {
            return ParamType(_m(0, 2), _m(1, 2), std::acos(_m(0,0)));
        }
        
        /** Set warp parameters */
        void setParameters(const ParamType &p) {
            _m(0, 2) = p(0, 0);
            _m(1, 2) = p(1, 0);
            
            float c = std::cos(p(2, 0));
            float s = std::sin(p(2, 0));
            
            _m(0,0) = c;
            _m(0,1) = -s;
            _m(1,0) = s;
            _m(1,1) = c;
        }
        
        /** 
            Compute the jacobian of the warp.
         
            The Jacobian matrix contains the partial derivatives of the warp parameters
            with respect to x and y coordinates, evaluated at the current value of parameters.
            In this case:
         
                    tx   ty   theta
                x   1    0  -sx - cy
                y   0    1   cx - sy
         
            with:
         
                c = cos(theta)
                s = sin(theta)
         */
        JacobianType jacobian(const cv::Point2f &p) const {
            JacobianType j = JacobianType::zeros();
            j(0, 0) = 1.f;
            j(1, 1) = 1.f;
            
            float c = _m(1, 1); // cos(theta)
            float s = _m(1, 0); // sin(theta)
            
            j(0, 2) = -s * p.x - c * p.y;
            j(1, 2) = c * p.x - s * p.y;
            
            return j;
        }
    };
    
    /**
        Warp implementation for Similarity motion.
     
        A similarity transform consists rotation, scale and translation. It preserves angles, 
        parallel lines and straight lines.
     
        The warp is parametrized with 4 parameters (tx, ty, a, b). In matrix notation
     
            (1 + a)    -b     tx
               b     (1 + a)  ty
     
            a = s * cos(theta)
            b = s * sin(theta)
     
        While this representation seems strange - one would rather expect the four
        parameters to be (tx, ty, theta and scale) - the chosen representation simplifies Jacobians,
        and matrix to parameter decomposition.
     
        Also note that in this parametrization a and b are not independent parameters.
     */
    template<>
    class Warp<WARP_SIMILARITY> : public PlanarWarp {
    public:
        
        typedef typename WarpTraits<WARP_SIMILARITY>::ParamType ParamType;
        typedef typename WarpTraits<WARP_SIMILARITY>::JacobianType JacobianType;
        
        /** Get warp parameters */
        ParamType getParameters() const {
            return ParamType(_m(0, 2), _m(1, 2), _m(0, 0) - 1.f, _m(1, 0));
        }
        
        /** Set warp parameters */
        void setParameters(const ParamType &p) {
            _m(0, 2) = p(0, 0);
            _m(1, 2) = p(1, 0);
            
            float a = p(2, 0);
            float b = p(3, 0);
            
            _m(0,0) = 1.f + a;
            _m(0,1) = -b;
            _m(1,0) = b;
            _m(1,1) = 1.f + a;
        }
        
        /** 
            Set warp parameters in canonical representation.
         
            As the warp is parametrized using variables a and b, which are both products
            of trigonometric functions and scaling, this method is provided to set the parameters
            using a more user friendly notation.
         
            \param p Parameters in canonical form (tx, ty, theta, scale)
         
            */
        void setParametersInCanonicalRepresentation(const ParamType &p) {
            setParameters(ParamType(p(0,0), p(1, 0), p(3,0) * std::cos(p(2,0)) - 1.f, p(3,0) * std::sin(p(2,0))));
        }
        
        /**
            Get warp parameters in canonical representation.
         
            As the warp is parametrized using variables a and b, which are both products
            of trigonometric functions and scaling, this method is provided to get the parameters
            using a more user friendly notation.
         
            \return Parameters in canonical form (tx, ty, theta, scale)
         
         */
        ParamType getParametersInCanonicalRepresentation()
        {
            // See http://math.stackexchange.com/questions/13150
            
            ParamType p;
            p(0,0) = _m(0, 2);
            p(1,0) = _m(1, 2);
            
            p(2,0) = std::atan2(-_m(0,1), _m(0,0));
            p(3,0) = std::sqrt(_m(0,0) * _m(0,0) + _m(0,1) * _m(0, 1)); // assume positive scaling factors
            
            
            return p;
        }
        
        /**
            Compute the jacobian of the warp.
         
            The Jacobian matrix contains the partial derivatives of the warp parameters
            with respect to x and y coordinates, evaluated at the current value of parameters.
            In this case:
         
                    tx   ty  a   b
                x   1    0   x  -y
                y   0    1   y   x
         
         */
        JacobianType jacobian(const cv::Point2f &p) const {
            JacobianType j = JacobianType::zeros();
            j(0, 0) = 1.f;
            j(1, 1) = 1.f;
            
            j(0, 2) = p.x;
            j(1, 2) = p.y;
            
            j(0, 3) = -p.y;
            j(1, 3) = p.x;
            
            return j;
        }
        
    };


    /**
        Warp an image using bilinear interpolation.
     
        This method warps a given source image onto a given destination image. It assumes the direction
        of the warp is such that for given pixel in the destination image, the warp reports the corresponding
        pixel in the source image.
     
        This method will call create on the destination image.
     
        \param src_ Source image
        \param dst_ Destination image
        \param dstSize Size of destination image
        \param scaleUp Optional scale up factor. Used with different levels of hierarchy
        \param scaleDown Optional scale down factor. Used with different levels of hierarchy
        \param w Warp function
     */
    template<class ChannelType, int SampleMethod, int WarpType>
    void warpImage(cv::InputArray src_, cv::OutputArray dst_, cv::Size dstSize, const Warp<WarpType> &w, const Sampler<SampleMethod> &s = Sampler<SampleMethod>(),float scaleUp = 1.f, float scaleDown = 1.f)
    {
        CV_Assert(src_.channels() == 1);
        
        dst_.create(dstSize, src_.type());
        
        cv::Mat src = src_.getMat();
        cv::Mat dst = dst_.getMat();
        
        for (int y = 0; y < dstSize.height; ++y) {
            ChannelType *r = dst.ptr<ChannelType>(y);
            
            for (int x = 0; x < dstSize.width; ++x) {
                cv::Point2f wp = w(cv::Point2f(x + 0.5f, y + 0.5f) * scaleUp) * scaleDown;
                r[x] = s.template sample<ChannelType>(src, wp);
            }
        }
    }
    
    
    
    
    
    
}

#endif