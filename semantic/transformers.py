
import semantic.transforms as transforms
from scipy.stats import norm
from scipy.stats import gamma as Gamma
import math
import numpy as np
import torch
from statsmodels.stats.proportion import proportion_confint

EPS = 1e-6

class AbstractTransformer:

    def process(self, inputs):
        raise NotImplementedError

    def calc_radius(self, pABar: float) -> float:
        return 0.0


class NoiseTransformer(AbstractTransformer):

    def __init__(self, sigma):
        super(NoiseTransformer, self).__init__()
        self.sigma = sigma
        self.noise_adder = transforms.Noise(self.sigma)

    def process(self, inputs):
        outs = self.noise_adder.batch_proc(inputs)
        return outs

    def calc_radius(self, pABar: float):
        radius = self.sigma * norm.ppf(pABar)
        return radius


class PointCloudRotation(AbstractTransformer):
    def __init__(self, sigma, canopy, axis='z'):
        super(PointCloudRotation, self).__init__()
        self.sigma = sigma
        self.rotation_adder3d = transforms.Rotation3D(canopy, sigma, axis)
        self.axis = axis
    
    def process(self, inputs):
        outs = self.rotation_adder3d.batch_proc(inputs)
        return outs

    def calc_radius(self, pABar: float):
        return self.sigma * norm.ppf(pABar)


class PointCloudShear(AbstractTransformer):
    def __init__(self, sigma, canopy, axis='z'):
        super(PointCloudShear, self).__init__()
        self.sigma = sigma
        self.shear_adder = transforms.Shear(canopy, sigma, axis)
        self.axis = axis

    def process(self, inputs):
        outs = self.shear_adder.batch_proc(inputs)
        return outs

    def calc_radius(self, pABar: float):
        return self.sigma * norm.ppf(pABar)


class PointCloudTwist(AbstractTransformer):
    def __init__(self, sigma, canopy, axis='z'):
        super(PointCloudTwist, self).__init__()
        self.sigma = sigma
        self.twist_adder = transforms.Twist(canopy, sigma, axis)
        self.axis = axis

    def process(self, inputs):
        outs = self.twist_adder.batch_proc(inputs)
        return outs

    def calc_radius(self, pABar: float):
        return self.sigma * norm.ppf(pABar)

class PointCloudTaper(AbstractTransformer):
    def __init__(self, canopy, angle = 0, axis='z'):
        super(PointCloudTaper, self).__init__()
        self.taper_adder = transforms.Taper(canopy, angle, axis)
        self.axis = axis
    
    def process(self, input):
        outs = self.taper_adder.batch_proc(input)
        return outs
    
    def calc_radius(self, pABar: float):
        return 1e+99

        
class PointCloudNoise(AbstractTransformer):
    def __init__(self, sigma):
        super(PointCloudNoise, self).__init__()
        self.sigma = sigma
        self.noise_adder = transforms.Noise(sigma)

    def process(self, inputs):
        outs = self.noise_adder.batch_proc(inputs)
        return outs

    def calc_radius(self, pABar: float):
        return self.sigma * norm.ppf(pABar)


class TaperNoiseTransformer(AbstractTransformer):
    def __init__(self, sigma, canopy, angle):
        super(TaperNoiseTransformer, self).__init__()
        self.sigma = sigma
        self.noise_adder = transforms.Noise(self.sigma)
        self.taper_adder = transforms.Taper(canopy, angle)

    def process(self, inputs):
        outs = inputs
        outs = self.taper_adder.batch_proc(outs)
        outs = self.noise_adder.batch_proc(outs)
        return outs

    def calc_radius(self, pABar: float):
        return 1e+99


class PointRotationNoiseTransformer(AbstractTransformer):
    def __init__(self, sigma, canopy, angle):
        super(PointRotationNoiseTransformer, self).__init__()
        self.sigma = sigma
        self.noise_adder = transforms.Noise(self.sigma)
        self.rotation_adder = transforms.GeneralRotation3D(canopy, angle)
    
    def process(self, inputs):
        outs = inputs
        outs = self.rotation_adder.batch_proc(outs)
        outs = self.noise_adder.batch_proc(outs)
        return outs

    def calc_radius(self, pABar: float):
        return 1e+99

class PointTwistRotationZ(AbstractTransformer):
    def __init__(self, sigma_t, sigma_r, canopy):
        super(PointTwistRotationZ,self).__init__()
        self.sigma_t = sigma_t
        self.sigma_r = sigma_r
        self.twist_adder = transforms.Twist(canopy, self.sigma_t, 'z')
        self.rotation_adder = transforms.Rotation3D(canopy, self.sigma_r, 'z')
    
    def process(self, inputs):
        outs = inputs
        outs = self.twist_adder.batch_proc(outs)
        outs = self.rotation_adder.batch_proc(outs)
        return outs
    
    def calc_radius(self, pABar: float):
        # Return a relative radius of an ellipse
        return norm.ppf(pABar)
    
class PointTaperRotationNoise(AbstractTransformer):
    def __init__(self, taper_scale, rotation_angle, noise_sd, canopy):
        super(PointTaperRotationNoise, self).__init__()
        self.taper_scale = taper_scale
        self.rotation_angle = rotation_angle
        self.noise_sd = noise_sd
        self.taper_adder = transforms.Taper(canopy, taper_scale)
        self.rotation_adder = transforms.Rotation3D(canopy, rotation_angle, 'z')
        self.noise_adder = transforms.Noise(self.noise_sd)
    
    def process(self, inputs):
        outs = inputs
        outs = self.rotation_adder.batch_proc_uniform(outs)
        outs = self.taper_adder.batch_proc(outs)
        outs = self.noise_adder.batch_proc(outs)
        return outs
    
    def calc_radius(self, pABar: float):
        return 1e+99



class PointTwistTaperRotationNoise(AbstractTransformer):
    def __init__(self, taper_scale, rotation_angle, twist_angle, noise_sd, canopy):
        super(PointTwistTaperRotationNoise, self).__init__()
        self.taper_scale = taper_scale
        self.rotation_angle = rotation_angle
        self.noise_sd = noise_sd
        self.taper_adder = transforms.Taper(canopy, taper_scale)
        self.twist_adder = transforms.Twist(canopy, twist_angle, 'z')
        self.rotation_adder = transforms.Rotation3D(canopy, rotation_angle, 'z')
        self.noise_adder = transforms.Noise(self.noise_sd)
    
    def process(self, inputs):
        outs = inputs
        outs = self.rotation_adder.batch_proc_uniform(outs)
        outs = self.taper_adder.batch_proc(outs)
        outs = self.twist_adder.batch_proc_uniform(outs)
        outs = self.noise_adder.batch_proc(outs)
        return outs
    
    def calc_radius(self, pABar: float):
        return 1e+99


class PointCloudLinear(AbstractTransformer):
    def __init__(self, sigma, canopy):
        super(PointCloudLinear, self).__init__()
        self.sigma = sigma
        self.linear_adder = transforms.Linear(canopy, sigma)
        self.canopy = canopy
    
    def process(self, inputs):
        outs = inputs
        outs = self.linear_adder.batch_proc(outs)
        return outs

    def monte_carlo(self, pA, A):
        k = np.zeros([3,3])
        for i in range(3):
            for j in range(3):
                k[i,j] = (1+A[j,j])**2 + np.linalg.norm(A[:,j]) ** 2 - A[j,j]**2
        k = np.sqrt(k).flatten()
        k = torch.from_numpy(k).cuda()
        num = 200000
        sample_points = torch.randn(num,9).cuda()
        radi = (sample_points ** 2 * (1 - 1/k**2)).sum(dim=1)
        radi2 = (sample_points ** 2 * (1-1/k**2) * (k**2)).sum(dim=1)
        for t1 in range(0,2000,1):
            t = (1000-t1)/50
            count = ((radi-t) < 0).sum().cpu()
            pA_high = proportion_confint(count, num, alpha = 0.01/2, method = 'normal')[1]
            if (pA_high >= pA):
                continue
            count_tilde = ((radi2-t) < 0).sum().cpu()
            p_tilde_low = proportion_confint(count_tilde, num, alpha = 0.01/2, method = 'normal')[0]
            return p_tilde_low
        return 0

    def test_certify(self, pA, alpha):
        # print(alpha)
        A = []
        pA_tilde = 1
        A.append(np.eye(3) * math.sqrt(alpha ** 2/3))
        A.append(-np.eye(3) * math.sqrt(alpha ** 2/3))
        A.append(np.diag(np.array([alpha, 0, 0])))
        A.append(np.diag(np.array([-alpha, 0, 0])))
        for i in A:
            pA_tilde = min(pA_tilde, self.monte_carlo(pA,i))
        if alpha < self.sigma * norm.ppf(pA_tilde) * (1 - alpha):
           return True
        return False



def gen_transformer(args, canopy) -> AbstractTransformer:
    if args.transtype == 'points-rotation':
        print(f'rotation point clouds {args.noise_sd}')
        print(f'axis {args.axis}')
        return PointCloudRotation(args.noise_sd, canopy, args.axis)
    elif args.transtype == 'points-shear':
        print(f'shear point clouds {args.noise_sd}')
        print(f'axis {args.axis}')
        return PointCloudShear(args.noise_sd, canopy, args.axis)
    elif args.transtype == 'points-twist':
        print(f'twist point clouds {args.noise_sd}')
        print(f'axis {args.axis}')
        return PointCloudTwist(args.noise_sd, canopy, args.axis)
    elif args.transtype == 'points-taper':
        print(f'taper point clouds {args.noise_sd}')
        print(f'axis {args.axis}')
        return PointCloudTaper(args.noise_sd, canopy, args.axis)
        # return PointCloudTaper(args.noise_sd, 0.8*args.noise_sd, canopy, args.axis)
    elif args.transtype == 'points-taper-noise':
        print(f'taper angle in +- {args.taper_angle} and noise in {args.noise_sd}')
        return TaperNoiseTransformer(args.noise_sd, canopy, args.taper_angle)
    elif args.transtype == 'points-noise':
        print(f'Gaussian noise {args.noise_sd}')
        return PointCloudNoise(args.noise_sd)
    elif args.transtype == 'points-rotation-noise':
        print(f'rotation angle in +- {args.rotation_angle} and noise in {args.noise_sd}')
        return PointRotationNoiseTransformer(args.noise_sd, canopy, args.rotation_angle)
    elif args.transtype == 'points-twist-rotationz':
        print(f'rotation angle in +- {args.rotation_angle} and twist in {args.noise_sd}')
        return PointTwistRotationZ(args.noise_sd, args.rotation_angle, canopy)
    elif args.transtype == 'points-taper-rotationz':
        print(f'rotation angle in +- {args.rotation_angle}, taper in +- {args.taper_angle} and noise in {args.noise_sd}')
        return PointTaperRotationNoise(args.taper_angle, args.rotation_angle, args.noise_sd, canopy)
    elif args.transtype == 'points-twist-taper-rotationz':
        print(f'rotation angle in +- {args.rotation_angle}, taper in +- {args.taper_angle}, twist in +- {args.twist_angle} and noise in {args.noise_sd}')
        return PointTwistTaperRotationNoise(args.taper_angle, args.rotation_angle, args.twist_angle, args.noise_sd, canopy)
    elif args.transtype == 'points-linear':
        print(f'linear transformation in += {args.noise_sd}')
        return PointCloudLinear(args.noise_sd, canopy)
    elif args.transtype == 'noise':
        print(f'noise {args.noise_sd}')
        return NoiseTransformer(args.noise_sd)
    else:
        raise NotImplementedError
