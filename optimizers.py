import numpy as np
import time

class Optimizer(object):
    def __init__(self, evaluate, initial_param_gen):
        self.evaluate = evaluate
        self.param_gen = initial_param_gen
        self.initial_param = initial_param_gen()
        self.D = len(self.initial_param)

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
        self.current = (self.evaluate(self.initial_param), self.initial_param)
        self.best = self.current
        self.best_time = 0

    def iterate(self):
        self.op = ""
        idxs = np.random.choice(range(self.D), size=1, replace=False)
        candidate = self.current[1].copy()
        candidate[idxs] = self.current[1][idxs] + np.random.rand(len(idxs))

#        candidate = np.minimum(1, np.maximum(0, candidate))
        while np.any(candidate < 0):
            outofrange = candidate < 0
            candidate[outofrange] += 1.
        while np.any(candidate > 1):
            outofrange = candidate > 1
            candidate[outofrange] -= 1.

        cand_eval = self.evaluate(candidate)
        prob = self.prob(self.t, self.current[0], cand_eval)
        self.op += "%.3f %.3f %.3f %.3f " % (cand_eval, self.current[0], self.temp(self.t), prob)

        if cand_eval < self.best[0]:
            a = np.linalg.norm(candidate - self.best[1])
            b = np.linalg.norm(candidate - self.current[1])
            self.op += ":-) (%.3f, %.3f) " % (a, b)
            self.best = (cand_eval, candidate)

        if np.random.rand() < prob:
            a = np.linalg.norm(candidate - self.best[1])
            b = np.linalg.norm(candidate - self.current[1])
            self.op += "new (%.3f, %.3f) " % (a, b)
            self.current = (cand_eval, candidate)
        
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
        simplex = [self.param_gen()]
        simplex.extend([np.roll(vec, i) + simplex[0] for i in idxs])
        self.simplex = [(self.evaluate(param), param) for param in simplex]


    def iterate(self):
        self.simplex.sort(key=lambda x: x[0])
        largest  = self.simplex[-1]  # worst performer
        secondlargest = self.simplex[-2]  # second worst performer
        smallest = self.simplex[0]
        centroid = np.mean([elem[1] for elem in self.simplex[:-1]], axis=0)

        candidate = centroid + 1*(centroid[1] - largest[1])  # expand in the direction opposite the worst performer
        candidate = (self.evaluate(candidate), candidate)

        if candidate[0] < smallest[0]:
            # that direction was a success, let's try to improve further
            newcandidate = centroid + 2*(candidate[1] - centroid[1])
            newcandidate = (self.evaluate(newcandidate), newcandidate)
            if newcandidate[0] < candidate[0]:
                self.op = "double extend"
                self.simplex[-1] = newcandidate
                return newcandidate
            else:
                self.op = "extend (1)"
                self.simplex[-1] = candidate
                return candidate

        if smallest[0] <= candidate[0] < secondlargest[0]:
            self.op = "extend (2)"
            self.simplex[-1] = candidate 
            return smallest

        if secondlargest[0] <= candidate[0] < largest[0]:
            # that direction wasn't a complete failure, let's try something more conservative
            self.op = "outer contract"
            newcandidate = centroid + 0.5*(candidate[1] - centroid[1])
            newcandidate = (self.evaluate(newcandidate), newcandidate)
        
        if largest[0] <= candidate[0]:
            self.op = "inner contract"
            newcandidate = centroid + 0.5*(largest[1] - centroid[1])
            newcandidate = (self.evaluate(newcandidate), newcandidate)

        if newcandidate[0] < largest[0]:
            self.simplex[-1] = newcandidate
            return newcandidate

        # form a new simplex closer to the best performer
        self.op = "all contract"
        newsimplex = [smallest[1] + 0.5*(param[1] - smallest[1]) for param in self.simplex]
        self.simplex = [(self.evaluate(param), param) for param in newsimplex]
        self.simplex.sort(key=lambda x: x[0])
        return self.simplex[0]