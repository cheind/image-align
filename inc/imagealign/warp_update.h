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

#ifndef IMAGE_ALIGN_WARP_UPDATE_H
#define IMAGE_ALIGN_WARP_UPDATE_H

#include <imagealign/warp.h>

namespace imagealign {
    
    /** 
        Default warp update for forward additive alignment algorithm.
     
        When using a custom warp that does not fit this method, make sure to provide an 
        overload of this method.
     
        \param w Warp to update.
        \param delta Delta parameters to add.
    */
    template<int WarpMode>
    void updateWarpForwardAdditive(Warp<WarpMode> &w, const typename WarpTraits<WarpMode>::ParamType &delta)
    {
        w.setParameters(w.getParameters() + delta);
    }
    
    /**
        Default warp update for forward compositional alignment algorithm.
        
        When using a custom warp that does not fit this method, make sure to provide an
        overload of this method.
     
        \param w Warp to update.
        \param delta Delta parameters to add.
     */
    template<int WarpMode>
    void updateWarpForwardCompositional(Warp<WarpMode> &w, const typename WarpTraits<WarpMode>::ParamType &delta)
    {
        Warp<WarpMode> wDelta = w; // Copy is done here when w has dynamic number of parameters
        wDelta.setIdentity();
        wDelta.setParameters(delta);
        w.setMatrix(w.getMatrix() * wDelta.getMatrix());
    }
    
    /**
        Default warp update for inverse compositional alignment algorithm.
     
        When using a custom warp that does not fit this method, make sure to provide an
        overload of this method.
     
        \param w Warp to update.
        \param delta Delta parameters to add.
     */
    template<int WarpMode>
    void updateWarpInverseCompositional(Warp<WarpMode> &w, const typename WarpTraits<WarpMode>::ParamType &delta)
    {
        Warp<WarpMode> wDelta = w; // Copy is done here when w has dynamic number of parameters
        wDelta.setIdentity();
        wDelta.setParameters(delta);
        w.setMatrix(w.getMatrix() * wDelta.getMatrix().inv());
    }
    
}

#endif