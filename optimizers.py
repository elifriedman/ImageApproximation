import numpy as np
import time

from collections import namedtuple

Param = namedtuple('Param', ['value', 'vector',])

def normalize(param):
    while np.any(param < 0):
        outofrange = param < 0
        param[outofrange] += 1.
    while np.any(param > 1):
        outofrange = param > 1
        param[outofrange] -= 1.
    return param


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


class ParticleSwarm(Optimizer):
    def __init__(self, evaluate, initial_param_gen, number_particles=10):
        super().__init__(evaluate, initial_param_gen)
        self.number_particles = number_particles
    
    def init(self):
        self.op = ""
        self.particles = []

        param = self.param_gen()
        self.best = Param(self.evaluate(param), param)
        for i in range(self.number_particles):
            pos = self.param_gen()
            particle = Particle(pos, self.evaluate)
            self.particles.append(particle)
            if particle.best.value < self.best.value:
                self.best = particle.best
        self.idxs = np.arange(self.number_particles)
        np.random.shuffle(self.idxs)  # update particles in random order
        self.ct = 0


    def iterate(self):
        idx = self.idxs[self.ct % self.number_particles]
        self.op = ""
        best = self.particles[idx].update(self.best)
        if best.value < self.best.value:
            self.best = best

        self.ct += 1
        if self.ct % self.number_particles == 0:
            np.random.shuffle(self.idxs)
        return self.best

class Particle(object):
    def __init__(self, initial_pos, eval_fn):
        self.pos = initial_pos
        self.vel = np.random.uniform(-1., 1, len(initial_pos))
        self.eval_fn = eval_fn
        self.best = Param(self.eval_fn(initial_pos), initial_pos.copy())

    def update(self, global_best):
        pbest_weight = np.random.uniform(0, 1, len(self.pos))
        gbest_weight = np.random.uniform(0, 1, len(self.pos))
        bp = self.best.vector - self.pos
        gp = global_best.vector - self.pos

        omega = 0.5 / np.log(2.)
        c1 = 0.5 + np.log(2.)
        c2 = 0.5 + np.log(2.)

        self.vel = omega*self.vel + c1*pbest_weight*bp + c2*gbest_weight*gp
        self.pos += self.vel
        self.pos = np.maximum(0., np.minimum(1., self.pos))

        value = self.eval_fn(self.pos)
        if value < self.best.value:
            self.best = Param(value, self.pos.copy())
        return self.best


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
        candidate = normalize(candidate)

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