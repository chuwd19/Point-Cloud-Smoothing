import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil, sqrt
from statsmodels.stats.proportion import proportion_confint
from semantic.transformers import AbstractTransformer
from typing import Union

EPS = 1e-6

class SemanticSmooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, transformer: AbstractTransformer):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.transformer = transformer

    def certify(self, x: torch.tensor, n0: int, maxn: int, alpha: float, batch_size: int, cAHat=None, margin_sq=None) -> Union[int, float]:
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """

        self.base_classifier.eval()
        if cAHat is None:
            # draw samples of f(x+ epsilon)
            counts_selection = self._sample_noise(x, n0)
            # use these samples to take a guess at the top class
            cAHat = counts_selection.argmax().item()
        nA, n = 0, 0
        pABar = 0.0
        while n < maxn:
            now_batch = min(batch_size, maxn - n)
            # draw more samples of f(x + epsilon)
            counts_estimation = self._sample_noise(x, now_batch)
            # print(counts_estimation)
            n += now_batch
            # use these samples to estimate a lower bound on pA
            nA += counts_estimation[cAHat].item()
            pABar = self._lower_confidence_bound(nA, n, alpha)
            r = self.transformer.calc_radius(pABar)
            # early stop if margin_sq is specified
            if margin_sq is not None and r >= sqrt(margin_sq):
                return cAHat, r - sqrt(margin_sq)
        if margin_sq is None:
            if r <= EPS:
                return SemanticSmooth.ABSTAIN, 0.0
            else:
                return cAHat, r
        else:
            return (SemanticSmooth.ABSTAIN if r <= EPS else cAHat), r - sqrt(margin_sq)



    def predict(self, x: torch.tensor, n0: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()

        n = 0
        counts = None
        while n < n0:
            now_batch = min(batch_size, n0 - n)
            # draw more samples of f(x + epsilon)
            counts_estimation = self._sample_noise(x, now_batch)
            # print(counts_estimation)
            n += now_batch
            if counts is None:
                counts = counts_estimation
            else:
                counts += counts_estimation
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return SemanticSmooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            # batch = x.repeat((num, 1, 1, 1))
            # print(x.shape, num)
            batch = x.repeat((num,1,1))
            batch_noised = self.transformer.process(batch).to(x.device)
            predictions = self.base_classifier(batch_noised).argmax(1)
            counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]


class SemanticSmoothSegmentation(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, transformer: AbstractTransformer, num_points : int):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.num_points = num_points
        self.transformer = transformer

    def certify(self, x: torch.tensor, n0: int, maxn: int, alpha: float, batch_size: int, cAHat=None, margin_sq=None) -> Union[int, float]:
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """

        predictions = np.zeros(self.num_points)
        radius = np.zeros(self.num_points)
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax(1)
        # print(cAHat)
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, maxn)
        for i in range(self.num_points):
            # use these samples to estimate a lower bound on pA
            nA = counts_estimation[i][cAHat[i]].item()
            pABar = self._lower_confidence_bound(nA, maxn, alpha)
            r = self.transformer.calc_radius(pABar)
            if r <= EPS:
                predictions[i] = SemanticSmoothSegmentation.ABSTAIN
                radius[i] = 0
            else:
                predictions[i] = cAHat[i]
                radius[i] = r
        return predictions, radius
        
    def predict(self, x: torch.tensor, n0: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()

        n = 0
        counts = None
        while n < n0:
            now_batch = min(batch_size, n0 - n)
            # draw more samples of f(x + epsilon)
            counts_estimation = self._sample_noise(x, now_batch)
            # print(counts_estimation)
            n += now_batch
            if counts is None:
                counts = counts_estimation
            else:
                counts += counts_estimation
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return SemanticSmooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros([self.num_points,self.num_classes],dtype=int)
            # batch = x.repeat((num, 1, 1, 1))
            # print(x.shape, num)
            batch = x.repeat((num,1,1))
            batch_noised = self.transformer.process(batch).to(x.device)
            predictions = self.base_classifier(batch_noised)
            # print(predictions.shape)
            prediction = predictions.max(1)[1]
            # print(prediction.shape)
            counts += self._count_arr(prediction.cpu().numpy(), self.num_points, self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, num_points: int, length: int) -> np.ndarray:
        counts = np.zeros((num_points, length), dtype=int)
        for batch in arr:
            for i,idx in enumerate(batch):
                counts[i][idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]