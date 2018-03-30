import numpy as np
from layers import Layer
from utils.tools import *
import copy

"""
This file defines layer types that are commonly used for recurrent neural networks.
"""

class RNNCell(Layer):
    def __init__(self, in_features, units, name='rnn_cell', initializer=Guassian()):
        """
        # Arguments
            in_features: int, the number of inputs features
            units: int, the number of hidden units
            initializer: Initializer class, to initialize weights
        """
        super(RNNCell, self).__init__(name=name)
        self.trainable = True

        self.kernel = initializer.initialize((in_features, units))
        self.recurrent_kernel = initializer.initialize((units, units))
        self.bias = np.zeros(units)

        self.kernel_grad = np.zeros(self.kernel.shape)
        self.r_kernel_grad = np.zeros(self.recurrent_kernel.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """
        # Arguments
            inputs: [input numpy array with shape (batch, in_features), 
                    state numpy array with shape (batch, units)]

        # Returns
            outputs: numpy array with shape (batch, units)
        """
        #############################################################
        # code here)
        inputs[0][np.isnan(inputs[0])] = 0
        affine_output = inputs[1].dot(self.recurrent_kernel) + inputs[0].dot( self.kernel) + self.bias
        outputs = np.tanh(affine_output)
        #############################################################
        return outputs

    def backward(self, in_grads, inputs):
        """
        # Arguments
            in_grads: numpy array with shape (batch, units), gradients to outputs
            inputs: [input numpy array with shape (batch, in_features), 
                    state numpy array with shape (batch, units)], same with forward inputs

        # Returns
            out_grads: [gradients to input numpy array with shape (batch, in_features), 
                        gradients to state numpy array with shape (batch, units)]
        """
        #############################################################
        # code here
        outputs = self.forward(inputs)

        in_grads = in_grads * (1 - outputs ** 2)
        out_grads_h = np.dot(in_grads, self.recurrent_kernel.T)
        dx = np.dot(in_grads, self.kernel.T)
        self.r_kernel_grad = np.dot(inputs[1].T, in_grads)
        self.kernel_grad = np.dot(inputs[0].T, in_grads)
        self.b_grad = np.sum(in_grads, axis=0)
        out_grads = [dx, out_grads_h]
    #############################################################
        return out_grads

    def update(self, params):
        """Update parameters with new params
        """
        for k,v in params.items():
            if '/kernel' in k:
                self.kernel = v
            elif '/recurrent_kernel' in k:
                self.recurrent_kernel = v
            elif '/bias' in k:
                self.bias = v
        
    def get_params(self, prefix):
        """Return parameters and gradients
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer
            grads: dictionary, store gradients of this layer

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/kernel': self.kernel,
                prefix+':'+self.name+'/recurrent_kernel': self.recurrent_kernel,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/kernel': self.kernel_grad,
                prefix+':'+self.name+'/recurrent_kernel': self.r_kernel_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None


class RNN(Layer):
    def __init__(self, cell, h0=None, name='rnn'):
        """
        # Arguments
            cell: instance of RNN Cell
            h0: default initial state, numpy array with shape (units,)
        """
        super(RNN, self).__init__(name=name)
        self.trainable = True
        self.cell = cell
        if h0 is None:
            self.h0 = np.zeros_like(self.cell.bias)
        else:
            self.h0 = h0
        
        self.kernel = self.cell.kernel
        self.recurrent_kernel = self.cell.recurrent_kernel
        self.bias = self.cell.bias

        self.kernel_grad = np.zeros(self.kernel.shape)
        self.r_kernel_grad = np.zeros(self.recurrent_kernel.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """
        # Arguments
            inputs: input numpy array with shape (batch, time_steps, in_features), 

        # Returns
            outputs: numpy array with shape (batch, time_steps, units)
        """
        #############################################################
        # code here
        units = 0
        if self.h0.ndim == 1:
            units = self.h0.shape[0]
        else:
            _, units = self.h0.shape
        batch, time_steps, in_features = inputs.shape
        outputs = np.zeros(shape=(batch, time_steps, units))

        for cur_time in range(time_steps):
            if cur_time == 0:
                outputs[:, cur_time, :] = self.cell.forward([inputs[:, cur_time, :], self.h0])
            else:
                outputs[:, cur_time, :] = self.cell.forward([inputs[:, cur_time, :], outputs[:, cur_time - 1, :]])
            self.kernel = self.cell.kernel
            self.recurrent_kernel = self.cell.recurrent_kernel
            self.bias = self.cell.bias
        #############################################################
        return outputs

    def backward(self, in_grads, inputs):
        """
        # Arguments
            in_grads: numpy array with shape (batch, time_steps, units), gradients to outputs
            inputs: numpy array with shape (batch, time_steps, in_features), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, time_steps, in_features), gradients to inputs
        """
        #############################################################
        # code here
        inputs[0][np.isnan(inputs[0])] = 0
        batch, time_steps, units = in_grads.shape
        _, _, in_features = inputs.shape

        out_grads = np.zeros((batch, time_steps, in_features))

        dh_prev = np.zeros((batch, units))
        outputs = self.forward(inputs)
        for cur_time in reversed(range(time_steps)):
            result = self.cell.backward(in_grads[:, cur_time, :] + dh_prev, [inputs[:, cur_time, :], outputs[:, cur_time - 1, :]])
            out_grads[:, cur_time, :] = result[0]
            dh_prev = result[1]
            self.kernel_grad += self.cell.kernel_grad
            self.r_kernel_grad += self.cell.r_kernel_grad
            self.b_grad += self.cell.b_grad

        #############################################################
        return out_grads

    def update(self, params):
        """Update parameters with new params
        """
        for k,v in params.items():
            if '/kernel' in k:
                self.kernel = v
            elif '/recurrent_kernel' in k:
                self.recurrent_kernel = v
            elif '/bias' in k:
                self.bias = v
        
    def get_params(self, prefix):
        """Return parameters and gradients
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer
            grads: dictionary, store gradients of this layer

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/kernel': self.kernel,
                prefix+':'+self.name+'/recurrent_kernel': self.recurrent_kernel,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/kernel': self.kernel_grad,
                prefix+':'+self.name+'/recurrent_kernel': self.r_kernel_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None        


class BidirectionalRNN(Layer):
    """ Bi-directional RNN in Concatenating Mode
    """
    def __init__(self, cell, h0=None, hr=None, name='brnn'):
        """Initialize two inner RNNs for forward and backward processes, respectively

        # Arguments
            cell: instance of RNN Cell(D, H) for initializing the two RNNs
            h0: default initial state for forward phase, numpy array with shape (units,)
            hr: default initial state for backward phase, numpy array with shape (units,)
        """
        super(BidirectionalRNN, self).__init__(name=name)
        self.trainable = True
        self.forward_rnn = RNN(cell, h0, 'forward_rnn')
        self.backward_rnn = RNN(copy.deepcopy(cell), hr, 'backward_rnn')

    def _reverse_temporal_data(self, x, mask):
        """ Reverse a batch of sequence data

        # Arguments
            x: a numpy array of shape (batch, time_steps, units), e.g.
                [[x_0_0, x_0_1, ..., x_0_k1, Unknown],
                ...
                [x_n_0, x_n_1, ..., x_n_k2, Unknown, Unknown]] (x_i_j is a vector of dimension of D)
            mask: a numpy array of shape (batch, time_steps), indicating the valid values, e.g.
                [[1, 1, ..., 1, 0],
                ...
                [1, 1, ..., 1, 0, 0]]

        # Returns
            reversed_x: numpy array with shape (batch, time_steps, units)
        """
        num_nan = np.sum(~mask, axis=1)
        reversed_x = np.array(x[:, ::-1, :])
        for i in range(num_nan.size):
            reversed_x[i] = np.roll(reversed_x[i], x.shape[1]-num_nan[i], axis=0)
        return reversed_x

    def forward(self, inputs):
        """
        Forward pass for concatenating hidden vectors obtained from the RNN 
        processing on normal sentences and the RNN processing on reversed sentences.
        Outputs concatenate the two produced sequences.

        # Arguments
            inputs: input numpy array with shape (batch, time_steps, in_features), 

        # Returns
            outputs: numpy array with shape (batch, time_steps, units*2)
        """
        mask = ~np.any(np.isnan(inputs), axis=2)
        forward_outputs = self.forward_rnn.forward(inputs)
        backward_outputs = self.backward_rnn.forward(self._reverse_temporal_data(inputs, mask))
        outputs = np.concatenate([forward_outputs, self._reverse_temporal_data(backward_outputs, mask)], axis=2)
        return outputs

    def backward(self, in_grads, inputs):
        """
        # Arguments
            in_grads: numpy array with shape (batch, time_steps, units*2), gradients to outputs
            inputs: numpy array with shape (batch, time_steps, in_features), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, time_steps, in_features), gradients to inputs
        """
        #############################################################
        # code here
        mask = ~np.any(np.isnan(inputs), axis=2)
        batch, time_stamp, double_unit = in_grads.shape
        unit = int(double_unit / 2)
        in_grads_array = np.reshape(in_grads,  batch * time_stamp * double_unit)
        [in_grads_forward, in_grads_backward] = np.split(in_grads_array, 2)

        forward_outputs = self.forward_rnn.backward(in_grads_forward.reshape(batch, time_stamp, unit), inputs)
        backward_outputs = self.backward_rnn.backward(self._reverse_temporal_data(in_grads_backward.reshape(batch, time_stamp, unit), mask), inputs)
        #############################################################
        return forward_outputs

    def update(self, params):
        """Update parameters with new params
        """
        for k,v in params.items():
            if '/forward_kernel' in k:
                self.forward_rnn.kernel = v
            elif '/forward_recurrent_kernel' in k:
                self.forward_rnn.recurrent_kernel = v
            elif '/forward_bias' in k:
                self.forward_rnn.bias = v
            elif '/backward_kernel' in k:
                self.backward_rnn.kernel = v
            elif '/backward_recurrent_kernel' in k:
                self.backward_rnn.recurrent_kernel = v
            elif '/backward_bias' in k:
                self.backward_rnn.bias = v
        
    def get_params(self, prefix):
        """Return parameters and gradients
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer
            grads: dictionary, store gradients of this layer

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/forward_kernel': self.forward_rnn.kernel,
                prefix+':'+self.name+'/forward_recurrent_kernel': self.forward_rnn.recurrent_kernel,
                prefix+':'+self.name+'/forward_bias': self.forward_rnn.bias,
                prefix+':'+self.name+'/backward_kernel': self.backward_rnn.kernel,
                prefix+':'+self.name+'/backward_recurrent_kernel': self.backward_rnn.recurrent_kernel,
                prefix+':'+self.name+'/backward_bias': self.backward_rnn.bias
            }
            grads = {
                prefix+':'+self.name+'/forward_kernel': self.forward_rnn.kernel_grad,
                prefix+':'+self.name+'/forward_recurrent_kernel': self.forward_rnn.r_kernel_grad,
                prefix+':'+self.name+'/forward_bias': self.forward_rnn.b_grad,
                prefix+':'+self.name+'/backward_kernel': self.backward_rnn.kernel_grad,
                prefix+':'+self.name+'/backward_recurrent_kernel': self.backward_rnn.r_kernel_grad,
                prefix+':'+self.name+'/backward_bias': self.backward_rnn.b_grad
            }
            return params, grads
        else:
            return None