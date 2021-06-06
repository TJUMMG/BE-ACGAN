# BE-ACGAN
Photo-realistic residual bit-depth enhancement by advanced conditional GAN

# Instructions: 
   1) Install TensorFlow(GPU);
   2) Download BE-ACGAN model from [Baidu Drive](https://pan.baidu.com/s/1FtvA7KXOiUXurZbOCscBnA)(4m43) to `./model/`
   3) Run 4-8/test_4_8.py to recover 8-bit images from 4-bit versions.\
      Run 4-16/test_4_16.py to recover 16-bit images from 4-bit versions.\
      It will directly compress and reconstruct images from testdata/.
   3) Results output to results_48/ or results_416/.
   * The image size in the code needs to be changed.

*********************************************************************

If you use this code, please cite the following publication:\
__J.Zhang, Q.Dou, J.Liu, Y.Su, W.Sun, "BE-ACGAN: Photo-realistic residual bit-depth enhancement by advanced conditional GAN", to appear in Displays__
    
*********************************************************************

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

For a copy of the GNU General Public License, please see <http://www.gnu.org/licenses/>. 

*********************************************************************

Here we thanks Christian Ledig et al. who are authors of  "Photo-realistic single image super-resolution using a generative adversarial network", published in IEEE Conference of Computer Vision and Pattern Recognition, for referring to their outstanding work.

*********************************************************************
