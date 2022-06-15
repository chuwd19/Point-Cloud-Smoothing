
import os
import random
import math
import numpy as np
import torch
import torchvision
import PIL.Image
from torchvision.transforms import *
import torchvision.transforms.functional as TF
import cv2
import numba
from numba import jit

EPS = 1e-6

class Noise:
    def __init__(self, sigma):
        self.sigma = sigma

    def proc(self, input):
        noise = torch.randn_like(input) * self.sigma
        return input + noise

    def batch_proc(self, inputs):
        noise = torch.randn_like(inputs) * self.sigma
        return inputs + noise


class Rotation3D:

    def __init__(self, canopy, sigma, axis):
        self.sigma = sigma
        self.num_points = canopy.shape[0]
        self.axis = axis
    
    def gen_param(self):
        theta = torch.randn(1)
        return theta * self.sigma
    
    def gen_param_uniform(self):
        theta = random.uniform(-self.sigma, self.sigma)
        return theta

    def proc(self, input, angle):
        angle = angle*math.pi/180
        # assume input: num_points x 3
        cost = math.cos(angle)
        sint = math.sin(angle)
        if self.axis == 'z':
            rotation_matrix = torch.tensor([[cost,-sint,0],[sint,cost,0],[0,0,1]]).to(input.device)
        elif self.axis == 'x':
            rotation_matrix = torch.tensor([[1,0,0],[0,cost,-sint],[0,sint,cost]]).to(input.device)
        elif self.axis == 'y':
            rotation_matrix = torch.tensor([[cost, 0, sint],[0,1,0],[-sint, 0, cost]]).to(input.device)
        else:
            raise NotImplementedError
        # print(input.shape)
        return torch.matmul(rotation_matrix,input.unsqueeze(-1)).squeeze(-1)
    
    def batch_proc(self, inputs):
        # print(inputs.shape)
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            # print(outs[i].shape, inputs[i].shape)
            outs[i] = self.proc(inputs[i], *self.gen_param())
        return outs

    def batch_proc_uniform(self, inputs):
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs[i] = self.proc(inputs[i], self.gen_param_uniform())
        return outs

class GeneralRotation3D:

    def __init__(self, canopy, sigma):
        self.sigma = sigma
        self.num_points = canopy.shape[0]
    
    def gen_param(self):
        k = np.random.randn(3)
        norm = np.linalg.norm(k)
        k = k / norm
        alpha = random.uniform(-self.sigma, self.sigma)
        return np.append(k,alpha)
    
    def proc(self, input, vec):
        angle = vec[3]*math.pi/180
        mat = np.array([[0,     -vec[2], vec[1]],
                        [vec[2],      0,-vec[0]],
                        [-vec[1], vec[0],     0]])
        rotation_matrix = np.eye(3) + np.sin(angle) * mat + (1 - np.cos(angle))*np.dot(mat,mat)
        rotation_matrix = torch.from_numpy(rotation_matrix).to(input.device).float()
        # assume input: num_points x 3
        return torch.matmul(rotation_matrix,input.unsqueeze(-1)).squeeze(-1)
    
    # def raw_proc(self, inputs, vec):
    #     outs = self.proc(inputs, vec)
    #     return outs
    
    def raw_proc(self, inputs, mat):
        return torch.matmul(mat, input.unsqueeze(-1)).squeeze(-1)

    def batch_proc(self, inputs):
        # print(inputs.shape)
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs[i] = self.proc(inputs[i], self.gen_param())
        return outs
 
class Shear:

    def __init__(self, canopy, sigma,axis):
        self.sigma = sigma
        self.num_points = canopy.shape[0]
        self.axis = axis

    def gen_param(self):
        theta = torch.randn(2)
        return theta*self.sigma

    def proc(self, input, t0,t1):
        if self.axis == 'z':
            shear_matrix = torch.tensor([[1,0,t0],[0,1,t1],[0,0,1]]).to(input.device)
        elif self.axis == 'x':
            shear_matrix = torch.tensor([[1,0,0],[t0,1,0],[t1,0,1]]).to(input.device)
        elif self.axis == 'y':
            shear_matrix = torch.tensor([[1, t0, 0],[0,1,0],[0, t1, 1]]).to(input.device)
        else:
            raise NotImplementedError
        return torch.matmul(shear_matrix,input.unsqueeze(-1)).squeeze(-1)
        
    def batch_proc(self, inputs):
        # print(inputs.shape)
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            # print(outs[i].shape, inputs[i].shape)
            outs[i] = self.proc(inputs[i], *self.gen_param())
        return outs


class Twist:

    def __init__(self, canopy, sigma, axis):
        self.sigma = sigma
        self.num_points = canopy.shape[0]
        self.axis = axis
    
    def gen_param(self):
        theta = torch.randn(1)
        return theta * self.sigma
    
    def proc(self, input, t):
        # assume input: num_points x 3
        t = t * math.pi / 180
        costz = torch.cos(input[:,2]*t).to(input.device)
        sintz = torch.sin(input[:,2]*t).to(input.device)
        outs = torch.zeros(input.shape).to(input.device)
        outs[:,0] = input[:,0]*costz - input[:,1]*sintz
        outs[:,1] = input[:,0]*sintz + input[:,1]*costz
        outs[:,2] = input[:,2]
        # print(input.shape)
        return outs
    
    def batch_proc(self, inputs):
        # print(inputs.shape)
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            # print(outs[i].shape, inputs[i].shape)
            outs[i] = self.proc(inputs[i], *self.gen_param())
        return outs

    def raw_proc(self, inputs, theta:float):
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            # print(outs[i].shape, inputs[i].shape)
            outs[i] = self.proc(inputs[i], theta)
        return outs

    def gen_param_uniform(self):
        theta = random.uniform(-self.sigma, self.sigma)
        return theta

    def batch_proc_uniform(self, inputs):
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs[i] = self.proc(inputs[i], self.gen_param_uniform())
        return outs


class Taper:

    def __init__(self, canopy, theta, axis='z'):
        self.theta = theta
        self.num_points = canopy.shape[0]
        self.axis = axis
    
    def gen_param(self):
        # theta = torch.randn(2)
        # theta[0] *= self.sigma
        # theta [1] *= self.tau
        return random.uniform(-self.theta, self.theta)

    def proc(self, input, theta:float):
        # taperz = 1 + theta1 * input[:,2] + theta2 * input[:,2]*input[:,2]*self.t**2/(1- self.t*input[:,2])
        # taperz = torch.exp(theta * input[:,2])
        taperz = 1 + theta * input[:,2]
        out = torch.zeros_like(input)
        out[:,0] = taperz * input[:,0]
        out[:,1] = taperz * input[:,1]
        out[:,2] = input [:,2]
        return out
    
    def raw_proc(self, inputs, theta:float):
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs[i] = self.proc(inputs[i], theta)
        return outs

    def batch_proc(self, inputs):
        # print(inputs.shape)
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            # print(outs[i].shape, inputs[i].shape)
            outs[i] = self.proc(inputs[i], self.gen_param())
        return outs

class Linear:
    def __init__(self, canopy, theta):
        self.num_points = canopy.shape[0]
        self.theta = theta

    def gen_param(self):
        return torch.randn(3,3) * self.theta
        
    def proc(self, input, t):
        transformation_matrix = torch.eye(3) + t
        transformation_matrix = transformation_matrix.to(input.device)
        return torch.matmul(transformation_matrix,input.unsqueeze(-1)).squeeze(-1)
    
    def batch_proc(self, inputs):
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs[i] = self.proc(inputs[i], self.gen_param())
        return outs

