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

#include <imagealign/config.h>

IA_DISABLE_PRAGMA_WARN(4190)
IA_DISABLE_PRAGMA_WARN(4244)
#include <opencv2/core/core.hpp>
IA_DISABLE_PRAGMA_WARN_END
IA_DISABLE_PRAGMA_WARN_END

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
    
    /** 2D Affine motion. See Warp<WARP_AFFINE>. */
    const int WARP_AFFINE = 3;
    
    /** 2D Perspective motion. See Warp<WARP_PERSPECTIVE>. */
    const int WARP_PERSPECTIVE = 4;
    
    /** 
        Each warp needs to provide traits.
     
        These traits usually provide matrix types during alignment.
        In the following N corresponds to the number of parameters of the
        warp.
     */
    template<int W, class Scalar>
    struct WarpTraits {
        enum {
            WarpMode = W
        };
        /** Precision of floating point type */
        typedef Scalar ScalarType;
        
        /** Type to hold a 2 dimensional point */
        typedef void PointType;
        
        /** Type to hold parameters of warp. Matrix of size Nx1.*/
        typedef void ParamType;
        
        /** Type to hold Jacobian of warp. Matrix of size 2xN. */
        typedef void JacobianType;
        
        /** Type to hold Hessian matrix. Matrix of size NxN. */
        typedef void HessianType;
        
        /** Type to hold the steepest descent image for a single pixel. Matrix of size 1xN. */
        typedef void PixelSDIType;
    };
    
    /** 
        Default warp traits implementation for compile time known parameter sizes.
     */
    template<int W, int N, class Scalar>
    struct WarpTraitsForCompileTimeKnownParameterCount {
        enum {
            WarpMode = W,
            NParameters = N
        };
        
        /** Precision of floating point type */
        typedef Scalar ScalarType;
        
        /** Type to hold a 2 dimensional point */
        typedef cv::Matx<Scalar, 2, 1> PointType;
        
        /** Type to hold parameters of warp. */
        typedef cv::Matx<Scalar, NParameters, 1> ParamType;
        
        /** Type to hold Jacobian of warp. */
        typedef cv::Matx<Scalar, 2, NParameters> JacobianType;
        
        /** Type to hold Hessian matrix. */
        typedef cv::Matx<Scalar, NParameters, NParameters> HessianType;
        
        /** Type to hold the steepest descent image for a single pixel */
        typedef cv::Matx<Scalar, 1, NParameters> PixelSDIType;

    };
    
    /**
        Warp traits for translational motion.
     */
    template<class Scalar>
    struct WarpTraits<WARP_TRANSLATION, Scalar> : WarpTraitsForCompileTimeKnownParameterCount<WARP_TRANSLATION, 2, Scalar> {};
    
    /**
        Warp traits for Euclidean motion.
     */
    template<class Scalar>
    struct WarpTraits<WARP_EUCLIDEAN, Scalar> : WarpTraitsForCompileTimeKnownParameterCount<WARP_EUCLIDEAN, 3, Scalar> {};
    
    /**
        Warp traits for Similarity motion.
     */
    template<class Scalar>
    struct WarpTraits<WARP_SIMILARITY, Scalar> : WarpTraitsForCompileTimeKnownParameterCount<WARP_SIMILARITY, 4, Scalar> {};
    
    /**
        Interface declaration for warps.
     
        Custom warps should at least offer the given methods and constructors.
     */
    template<int WarpMode, class Scalar>
    class Warp {
    public:
        /** Access the traits type. */
        typedef WarpTraits<WarpMode, Scalar> Traits;
        
        /** Be copy-constructible from another warp. */
        Warp(const Warp<WarpMode, Scalar> &other);
        
        /** Be able to set to identity transform. */
        void setIdentity();
        
        /** Be able to warp single pair of image coordinates. */
        typename Traits::PointType operator()(const typename Traits::PointType &p) const;
        
        /** Be able to compute the Jacobian of the warp at a given coordinate pair. */
        typename Traits::JacobianType jacobian(const typename Traits::PointType &p) const;
        
        /** Be able to perform the forward additive step. Needed only when using AlignForwardAdditive. */
        void updateForwardAdditive(const typename Traits::ParamType &delta);
        
        /** Be able to perform the forward compositional step. Needed only when using AlignForwardCompositional. */
        void updateForwardCompositional(const typename Traits::ParamType &delta);
        
        /** Be able to perform the inverse compositional step. Needed only when using AlignInverseCompositional. */
        void updateInverseCompositional(const typename Traits::ParamType &delta);
        
    };

    /**
        Base class for warps based on planar motions.
     
        Planar motions in 2D can be well described by a 3x3 matrix, which is
        what this class does. It can thus provide a (not always) efficient warp
        method for image coordinates.
     */
    template<int WarpMode, class Scalar>
    class PlanarWarp {
    public:
        typedef cv::Matx<Scalar, 3, 3> MType;
        
        inline PlanarWarp() {
            setIdentity();
        }
        
        inline void setIdentity() {
            _m = MType::eye();
        }
        
        inline MType matrix() const {
            return _m;
        }
        
        inline void setMatrix(const MType &m) {
            _m = m;
        }
        
        inline MType invMatrix() const {
            if (WarpMode < WARP_PERSPECTIVE) {
                // Faster variant for Affine matrices
                cv::Matx<Scalar, 2, 2> ir = _m. template get_minor<2, 2>(0, 0).inv();
                cv::Matx<Scalar, 2, 1> ib = -ir * _m. template get_minor<2, 1>(0, 2);
                
                MType result;
                
                result(0, 0) = ir(0, 0);
                result(0, 1) = ir(0, 1);
                result(0, 2) = ib(0, 0);
                
                result(1, 0) = ir(1, 0);
                result(1, 1) = ir(1, 1);
                result(1, 2) = ib(1, 0);
                
                result(2, 0) = 0.f;
                result(2, 1) = 0.f;
                result(2, 2) = 1.f;
                
                return result;
                
            } else {
                return _m.inv();
            }
        }
        
        /** Warp point */
        inline cv::Matx<Scalar, 2, 1> operator()(const cv::Matx<Scalar, 2, 1> &p) const {
            
            cv::Matx<Scalar, 3, 1> x(p(0), p(1), 1.f);
            x = _m * x;
            
            // Compile time if to speed up matrix calculation.
            // Todo: Consider Affine matrix to be represented by 2x3
            if (WarpMode < WARP_PERSPECTIVE) {
                return x.template get_minor<2, 1>(0, 0);
            } else {
                x = x * (Scalar(1) / x(2, 0)); // Normalize
                return x.template get_minor<2, 1>(0, 0);
            }
            
        }
        
    protected:
        MType _m;
    };
    
    /** 
        Warp implementation for pure translational motion.
     
        A translational transform preserves orientation, lengths, angles, parallel lines 
        and straight lines.
     
        The warp is parametrized with 2 parameters (tx, ty). In matrix notation
     
            1  0  tx
            0  1  ty

    */
    template<class Scalar>
    class Warp<WARP_TRANSLATION, Scalar> : public PlanarWarp<WARP_TRANSLATION, Scalar> {
    private:
        using PlanarWarp<WARP_TRANSLATION, Scalar>::_m;
        using PlanarWarp<WARP_TRANSLATION, Scalar>::matrix;
        using PlanarWarp<WARP_TRANSLATION, Scalar>::setMatrix;

    public:
        typedef WarpTraits<WARP_TRANSLATION, Scalar> Traits;
        typedef typename Traits::PointType PointType;
        typedef typename Traits::ParamType ParamType;
        typedef typename Traits::JacobianType JacobianType;

        /** Get warp parameters */
        ParamType parameters() const {
            return ParamType(_m(0, 2), _m(1, 2));
        }
        
        /** Set warp parameters */
        void setParameters(const ParamType &p) {
            _m(0, 2) = p(0, 0);
            _m(1, 2) = p(1, 0);
        }
        
        /**
            Compute the jacobian of the warp.
            
            The Jacobian matrix contains the partial derivatives of the warp parameters
            with respect to x and y coordinates, evaluated at the given pixel position.
            In this case:
                
                   tx   ty
                x   1    0
                y   0    1
         
         */
        JacobianType jacobian(const PointType &p) const {
            JacobianType j = JacobianType::zeros();
            j(0, 0) = Scalar(1);
            j(1, 1) = Scalar(1);
            return j;
        }
        
        /** Forward additive step. */
        void updateForwardAdditive(const ParamType &delta) {
            setParameters(parameters() + delta);
        }
        
        /** Forward compositional step. */
        void updateForwardCompositional(const ParamType &delta) {
            Warp<WARP_TRANSLATION, Scalar> wDelta;
            wDelta.setParameters(delta);
            setMatrix(matrix() * wDelta.matrix());
        }
        
        /** Inverse compositional step. */
        void updateInverseCompositional(const ParamType &delta) {
            Warp<WARP_TRANSLATION, Scalar> wDelta;
            wDelta.setParameters(delta);
            setMatrix(matrix() * wDelta.invMatrix());
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
    template<class Scalar>
    class Warp<WARP_EUCLIDEAN, Scalar> : public PlanarWarp<WARP_EUCLIDEAN, Scalar> {
    private:
        using PlanarWarp<WARP_EUCLIDEAN, Scalar>::_m;
        using PlanarWarp<WARP_EUCLIDEAN, Scalar>::matrix;
        using PlanarWarp<WARP_EUCLIDEAN, Scalar>::setMatrix;
        
    public:
        
        typedef WarpTraits<WARP_EUCLIDEAN, Scalar> Traits;
        typedef typename Traits::PointType PointType;
        typedef typename Traits::ParamType ParamType;
        typedef typename Traits::JacobianType JacobianType;
        
        /** Get warp parameters */
        ParamType parameters() const {
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
        JacobianType jacobian(const PointType &p) const {
            JacobianType j = JacobianType::zeros();
            j(0, 0) = Scalar(1);
            j(1, 1) = Scalar(1);
            
            float c = _m(1, 1); // cos(theta)
            float s = _m(1, 0); // sin(theta)
            
            j(0, 2) = -s * p(0) - c * p(1);
            j(1, 2) = c * p(0) - s * p(1);
            
            return j;
        }
        
        /** Forward additive step. */
        void updateForwardAdditive(const ParamType &delta) {
            setParameters(parameters() + delta);
        }
        
        /** Forward compositional step. */
        void updateForwardCompositional(const ParamType &delta) {
            Warp<WARP_EUCLIDEAN, Scalar> wDelta;
            wDelta.setParameters(delta);
            setMatrix(matrix() * wDelta.matrix());
        }
        
        /** Inverse compositional step. */
        void updateInverseCompositional(const ParamType &delta) {
            Warp<WARP_EUCLIDEAN, Scalar> wDelta;
            wDelta.setParameters(delta);
            setMatrix(matrix() * wDelta.invMatrix());
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
    template<class Scalar>
    class Warp<WARP_SIMILARITY, Scalar> : public PlanarWarp<WARP_SIMILARITY, Scalar> {
    private:
        using PlanarWarp<WARP_SIMILARITY, Scalar>::_m;
        using PlanarWarp<WARP_SIMILARITY, Scalar>::matrix;
        using PlanarWarp<WARP_SIMILARITY, Scalar>::setMatrix;
        
    public:
        
        typedef WarpTraits<WARP_SIMILARITY, Scalar> Traits;
        typedef typename Traits::PointType PointType;
        typedef typename Traits::ParamType ParamType;
        typedef typename Traits::JacobianType JacobianType;
        
        /** Get warp parameters */
        ParamType parameters() const {
            return ParamType(_m(0, 2), _m(1, 2), _m(0, 0) - Scalar(1), _m(1, 0));
        }
        
        /** Set warp parameters */
        void setParameters(const ParamType &p) {
            _m(0, 2) = p(0, 0);
            _m(1, 2) = p(1, 0);
            
            float a = p(2, 0);
            float b = p(3, 0);
            
            _m(0,0) = Scalar(1) + a;
            _m(0,1) = -b;
            _m(1,0) = b;
            _m(1,1) = Scalar(1) + a;
        }
        
        /** 
            Set warp parameters in canonical representation.
         
            As the warp is parametrized using variables a and b, which are both products
            of trigonometric functions and scaling, this method is provided to set the parameters
            using a more user friendly notation.
         
            \param p Parameters in canonical form (tx, ty, theta, scale)
         
            */
        void setParametersInCanonicalRepresentation(const ParamType &p) {
            setParameters(ParamType(p(0,0), p(1, 0), p(3,0) * std::cos(p(2,0)) - Scalar(1), p(3,0) * std::sin(p(2,0))));
        }
        
        /**
            Get warp parameters in canonical representation.
         
            As the warp is parametrized using variables a and b, which are both products
            of trigonometric functions and scaling, this method is provided to get the parameters
            using a more user friendly notation.
         
            \return Parameters in canonical form (tx, ty, theta, scale)
         
         */
        ParamType parametersInCanonicalRepresentation()
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
        JacobianType jacobian(const PointType &p) const {
            JacobianType j = JacobianType::zeros();
            j(0, 0) = Scalar(1);
            j(1, 1) = Scalar(1);
            
            j(0, 2) = p(0);
            j(1, 2) = p(1);
            
            j(0, 3) = -p(1);
            j(1, 3) = p(0);
            
            return j;
        }
        
        /** Forward additive step. */
        void updateForwardAdditive(const ParamType &delta) {
            setParameters(parameters() + delta);
        }
        
        /** Forward compositional step. */
        void updateForwardCompositional(const ParamType &delta) {
            Warp<WARP_SIMILARITY, Scalar> wDelta;
            wDelta.setParameters(delta);
            setMatrix(matrix() * wDelta.matrix());
        }
        
        /** Inverse compositional step. */
        void updateInverseCompositional(const ParamType &delta) {
            Warp<WARP_SIMILARITY, Scalar> wDelta;
            wDelta.setParameters(delta);
            setMatrix(matrix() * wDelta.invMatrix());
        }
        
    };
    
    typedef Warp<WARP_TRANSLATION, float> WarpTranslationF;
    typedef Warp<WARP_TRANSLATION, double> WarpTranslationD;
    
    typedef Warp<WARP_EUCLIDEAN, float> WarpEuclideanF;
    typedef Warp<WARP_EUCLIDEAN, double> WarpEuclideanD;
    
    typedef Warp<WARP_SIMILARITY, float> WarpSimilarityF;
    typedef Warp<WARP_SIMILARITY, double> WarpSimilarityD;
    
}

#endif