from functions import Function
from numba import jit
import np_utils
import numpy as np

class Convolution2D(Function):

    @staticmethod
    @jit(nopython=True)
    def example_helper_func():
        """
        Example of an accelerated function, Notice the Numba jit decorator on top.
        """
        pass

    def forward(self, stride, padding, *args):
        """
        Forward pass of the convolution operation between two four dimensional tensors.
        :param stride: Convolution stride, defaults to 1.
        :param padding: Convolution padding, defaults to 0.
        :param args: Operands of convolution operation (input(batch_size, in_channels, H, W), kernel(out_channels, in_channels, Hk, Wk)).
        :return: Output of the convolution operation.
        """
        #TODO
        self.parents = args
        
        (batch_size, input_in_channels, H, W) = self.parents[0].value.shape
        input_image = self.parents[0].value
        
        (kernels, kernel_in_channels, Hk, Wk) = self.parents[1].value.shape
        kernel = self.parents[1].value
        
        output_channels = int((input_in_channels + 2 * padding - kernel_in_channels) / stride) + 1
        output_H = int((H + 2 * padding - Hk) / stride) + 1
        output_W = int((W + 2 * padding - Wk) / stride) + 1    
        
        @jit(nopython=True)
        def helper_forward():
            output_images = np.zeros((batch_size, output_channels*kernels, output_H, output_W))
            for i in range(batch_size):
                for channel in range(output_channels):
                    for h in range(output_H):
                        for w in range(output_W):
                            for j in range(kernels):
                                _image = input_image[i,channel:channel+kernel_in_channels,h:h+Hk,w:w+Wk]
                                _kernel = kernel[j,:,:,:]
                                output_images[i,j+channel,h,w] = np.sum(np.multiply(_image, _kernel))
            
            return output_images
        
        return helper_forward()   

    def backward(self, gradient):
        """
        Sets the gradients for operands of convolution operation.
        :param gradient: Upstream gradient.
        """
        #TODO
        (batch_size, input_in_channels, H, W) = self.parents[0].value.shape
        input_image = self.parents[0]
        
        (kernels, kernel_in_channels, Hk, Wk) = self.parents[1].value.shape
        kernel = self.parents[1]
        
        (gradients, channels, Hg, Wg) = gradient.shape
        
        for g in range(gradients):
            for i in range(kernels):
                for h in range(Hg):
                    for w in range(Wg):
                        _image = gradient[g, i, h, w]
                        input_image.grad[g,:,h:h+Hk,w:w+Wk] += kernel.value[i,:,:,:] * _image
                        kernel.grad[i,:,:,:] += input_image.value[g,:,h:h+Hk,w:w+Wk] * _image

class Reshape(Function):
    def forward(self, shape, *args):
        """
        Forward pass of the reshape operation on a tensor
        :param shape: tuple of required dimension.
        :param args: Input tensor to be reshaped.
        :return: reshaped tensor.
        """
        #TODO
        self.parents = list(args)
        return np.reshape(self.parents[0].value, shape)

    def backward(self, gradient):
        """
        Sets the gradient for input of reshape operation.
        :param gradient: Upstream gradient.
        """
        #TODO
        input_image = self.parents[0]
        input_image.grad += np.reshape(gradient, input_image.value.shape)
        