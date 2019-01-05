import numpy as np
import time

from collections import namedtuple

Param = namedtuple('Param', ['value', 'vector',])

class Optimizer(object):
    def __init__(self, evaluate, initial_param_gen):
        self.evaluate = evaluate
        self.param_gen = initial_param_gen
        initial_param = initial_param_gen()
        self.initial_param = Param(self.evaluate(initial_param), initial_param)
        self.D = len(self.initial_param.vector)

    def init(self):
        pass

    def iterate(self):
        pass


class SimulatedAnnealing(Optimizer):
    def __init__(self, evaluate, initial_param_gen):
        super().__init__(evaluate, initial_param_gen)
        gamma = 1.
        self.temp = lambda t: gamma / (t + 1)

    def prob(self, time, current, new):
        exp = np.exp(-(new - current)/self.temp(time))
        return min(1, exp)

    def init(self):
        self.t = 0
        self.current = self.initial_param
        self.best = self.current
        self.best_time = 0

    def iterate(self):
        self.op = ""
        idxs = np.random.choice(range(self.D), size=1, replace=False)
        candidate = self.current.vector.copy()
        candidate[idxs] = self.current.vector[idxs] + np.random.rand(len(idxs))

#        candidate = np.minimum(1, np.maximum(0, candidate))
        while np.any(candidate < 0):
            outofrange = candidate < 0
            candidate[outofrange] += 1.
        while np.any(candidate > 1):
            outofrange = candidate > 1
            candidate[outofrange] -= 1.

        cand_eval = self.evaluate(candidate)
        prob = self.prob(self.t, self.current.value, cand_eval)
        self.op += "%.3f %.3f %.3f %.3f " % (cand_eval, self.current.value, self.temp(self.t), prob)

        if cand_eval < self.best.value:
            a = np.linalg.norm(candidate - self.best.vector)
            b = np.linalg.norm(candidate - self.current.vector)
            self.op += ":-) (%.3f, %.3f) " % (a, b)
            self.best = Param(cand_eval, candidate)

        if np.random.rand() < prob:
            a = np.linalg.norm(candidate - self.best.vector)
            b = np.linalg.norm(candidate - self.current.vector)
            self.op += "new (%.3f, %.3f) " % (a, b)
            self.current = Param(cand_eval, candidate)
        
        self.t += 1
        return self.best

class SimplexOptimizer(Optimizer):
    def __init__(self, evaluate, initial_param_gen):
        super().__init__(evaluate, initial_param_gen)

    def init(self):
        vec = np.zeros(self.D)
        vec[0] = .5
        idxs = [i for i in range(self.D)]
        np.random.shuffle(idxs)
        idxs = idxs
        simplex = [self.initial_param.vector]
        simplex.extend([np.roll(vec, i) + simplex[0] for i in idxs])
        self.simplex = [Param(self.evaluate(param), param) for param in simplex]


    def iterate(self):
        self.simplex.sort(key=lambda x: x.value)
        largest  = self.simplex[-1]  # worst performer
        secondlargest = self.simplex[-2]  # second worst performer
        smallest = self.simplex.value
        centroid = np.mean([elem.vector for elem in self.simplex[:-1]], axis=0)

        candidate = centroid + 1*(centroid.vector - largest.vector)  # expand in the direction opposite the worst performer
        candidate = (self.evaluate(candidate), candidate)

        if candidate.value < smallest.value:
            # that direction was a success, let's try to improve further
            newcandidate = centroid + 2*(candidate.vector - centroid.vector)
            newcandidate = (self.evaluate(newcandidate), newcandidate)
            if newcandidate.value < candidate.value:
                self.op = "double extend"
                self.simplex[-1] = newcandidate
                return newcandidate
            else:
                self.op = "extend (1)"
                self.simplex[-1] = candidate
                return candidate

        if smallest.value <= candidate.value < secondlargest.value:
            self.op = "extend (2)"
            self.simplex[-1] = candidate 
            return smallest

        if secondlargest.value <= candidate.value < largest.value:
            # that direction wasn't a complete failure, let's try something more conservative
            self.op = "outer contract"
            newcandidate = centroid + 0.5*(candidate.vector - centroid.vector)
            newcandidate = (self.evaluate(newcandidate), newcandidate)
        
        if largest.value <= candidate.value:
            self.op = "inner contract"
            newcandidate = centroid + 0.5*(largest.vector - centroid.vector)
            newcandidate = (self.evaluate(newcandidate), newcandidate)

        if newcandidate.value < largest.value:
            self.simplex[-1] = newcandidate
            return newcandidate

        # form a new simplex closer to the best performer
        self.op = "all contract"
        newsimplex = [smallest.vector + 0.5*(param.vector - smallest.vector) for param in self.simplex]
        self.simplex = [(self.evaluate(param), param) for param in newsimplex]
        self.simplex.sort(key=lambda x: x.value)
        return self.simplex.value