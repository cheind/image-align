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

#include <opencv2/core/core.hpp>


namespace imagealign {
    
    /** Forwards-additive image alignment.
        
        'Best' aligns a template image with a target image through minimization of
        the sum of squared intensity errors between the warped target image and the
        template image with respect to the warp parameters.
     
        This algorithm is the classic algorithm proposed by Lucas-Kanade. Baker
        and Matthews coined it later the forwards-additive algorithm because of 
        its properties:
         - forward: direction of warp parameter estimation
         - additive: update of warp parameters happens through summing of 
                     warp parameter increments.
     
        ## Based on
     
        Lucas, Bruce D., and Takeo Kanade. 
        "An iterative image registration technique with an application to stereo vision." 
        IJCAI. Vol. 81. 1981.
     
        Baker, Simon, and Iain Matthews. 
        "Lucas-kanade 20 years on: A unifying framework." 
        International journal of computer vision 56.3 (2004): 221-255.

     */
     
    void alignForwardsAdditive(cv::InputArray tmpl, cv::InputArray target);
    
    
}

#endif