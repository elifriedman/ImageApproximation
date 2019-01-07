import numpy as np
import time
import logging

from collections import namedtuple

logger = logging.getLogger("optimizer")
Param = namedtuple('Param', ['value', 'vector',])

def normalize(param):
    return param % 1.
#    return np.maximum(0., np.minimum(1., param))


class Optimizer(object):
    def __init__(self, evaluate, initial_param_gen):
        self.evaluate = evaluate
        self.param_gen = initial_param_gen
        initial_param = initial_param_gen()
        self.initial_param = Param(self.evaluate(initial_param), initial_param)
        self.D = len(self.initial_param.vector)
        self.iteration = -1

    def init(self):
        pass

    def iterate(self):
        self.iteration += 1


class ParticleSwarm(Optimizer):
    def __init__(self, evaluate, initial_param_gen, number_particles=40,
                 w_omega = 0.5 / np.log(2),
                 w_pb = 0.5 + np.log(2),
                 w_gb = 0.5 + np.log(2),
                 randomize_order=True, **kwargs):
        """Particle Swarm Optimizer

        Parameters
        ----------
        number_particles : int
            Number of particles in the swarm
        w_omega : float
            Weight to multipliy the current velociy
        w_pb : float
            Weight to multiply the component of the velocity in the direction of the personal best point
        w_gb : float
            Weight to multiply the component of the velocity in the direction of the global best point
        randomize_order : bool
            Whether to randomize order in which particles are updated
        """
        super().__init__(evaluate, initial_param_gen)
        self.number_particles = number_particles
        self.particles = []
        self.w_omega = w_omega
        self.w_pb = w_pb
        self.w_gb = w_gb
        self.randomize_order = randomize_order
    
    def init(self):

        param = self.param_gen()
        self.best = Param(self.evaluate(param), param)
        for i in range(self.number_particles):
            pos = self.param_gen()
            particle = Particle(pos, self.evaluate, self.w_omega, self.w_pb, self.w_gb)
            self.particles.append(particle)
            if particle.best.value < self.best.value:
                self.best = particle.best
        self.idxs = np.arange(self.number_particles)
        if self.randomize_order:
            np.random.shuffle(self.idxs)  # update particles in random order


    def iterate(self):
        super().iterate()
        if len(self.particles) == 0:
            self.init()
        idx = self.idxs[(self.iteration - 1) % self.number_particles]
        best = self.particles[idx].update(self.best)
        if best.value < self.best.value:
            self.best = best

        if self.iteration % self.number_particles == 0 and self.randomize_order:
            np.random.shuffle(self.idxs)
        return self.best

class Particle(object):
    def __init__(self, initial_pos, eval_fn,
                 w_omega = 0.5 / np.log(2),
                 w_pb = 0.5 + np.log(2),
                 w_gb = 0.5 + np.log(2), **kwargs):
        self.pos = initial_pos
        self.vel = np.random.uniform(-1., 1, len(initial_pos))
        self.eval_fn = eval_fn
        self.best = Param(self.eval_fn(initial_pos), initial_pos.copy())
        self.w_omega = w_omega
        self.w_pb = w_pb
        self.w_gb = w_gb

    def update(self, global_best):
        pbest_weight = np.random.uniform(0, 1, len(self.pos))
        gbest_weight = np.random.uniform(0, 1, len(self.pos))

        bp = self.best.vector - self.pos
        gp = global_best.vector - self.pos

        self.vel = self.w_omega*self.vel + self.w_pb*pbest_weight*bp + self.w_gb*gbest_weight*gp
        self.pos += self.vel
        self.pos = np.maximum(0., np.minimum(1., self.pos))

        value = self.eval_fn(self.pos)
        if value < self.best.value:
            self.best = Param(value, self.pos.copy())
        return self.best


class SimulatedAnnealing(Optimizer):
    def __init__(self, evaluate, initial_param_gen, temp_schedule=None,
                 w_gamma=1., accept_probability=None,
                 num_axes_to_update=1, **kwargs):
        """Simulated Annealing Optimizer

        Parameters
        ----------
        temp_schedule : function(int) -> float
            a function that takes as input the current iteration and outputs the desired temperature
            for that iteration. Defaults to gamma / log(iteration + 2)
        w_gamma : float
            initial multiplier for gamma function. Used in the default temp_schedule
        accept_probability : function(iteration [int], current_value [float], candidate_value [float]) -> float in [0, 1.]
            a function that outputs the probability of accepting a candidate point given the current iteration,
            the current point's value, and the candidate point's value.
            Defaults to min(1., exp(-(candidate_value - current_value) / temperature(iteration)))
        num_axes_to_update : int
            The number of axes of the current point that should be annealed per iteration
        """
        super().__init__(evaluate, initial_param_gen)
        self.temp = temp_schedule if temp_schedule else lambda t: w_gamma / np.log(t + 2)
        self.prob = accept_probability if accept_probability else self._prob
        self.num_axes = num_axes_to_update
        self.current = None

    def _prob(self, time, current, new):
        exp = np.exp(-(new - current)/self.temp(time))
        return min(1, exp)

    def init(self):
        self.t = 0
        self.current = self.initial_param
        self.best = self.current
        self.best_time = 0

    def iterate(self):
        if self.current is None:
            self.init()

        idxs = np.random.choice(range(self.D), size=self.num_axes, replace=False)
        candidate = self.current.vector.copy()
        candidate[idxs] = self.current.vector[idxs] + np.random.rand(len(idxs))

        candidate = normalize(candidate)
        candidate = Param(self.evaluate(candidate), candidate)

        if candidate.value < self.best.value:
            dist = np.linalg.norm(candidate.vector - self.best.vector)
            logger.debug("temp=%.3f, improved by=%.3f, distance from previous=%.3f",
                          self.temp(self.t), self.best.value - candidate.value, dist)
            self.best = candidate

        prob = self.prob(self.t, self.current.value, candidate.value)
        if np.random.rand() < prob:
            self.current = candidate
        
        self.t += 1
        return self.best

class SimplexOptimizer(Optimizer):
    def __init__(self, evaluate, initial_param_gen,
                 w_reflect=1., w_expand=2.,
                 w_contract=.5, w_shrink=.5,
                 random_init=False):
        """Nelder Mead Simplex Optimizer Algorithm

        Parameters
        ----------
        w_reflect : float
            weight for choosing a point that's opposite the worst performing point
        w_expand : float
            weight for choosing a point that's even farther opposite the worst performing point
        w_contract : float
            weight for choosing a point that's a bit closer to the worst performing point
        w_shrink : float
            how much to shrink each point in the simplex closer to the best performing point
        random_init : boolean
            whether to initialize the simplex to include orthogonal points (False) or random points (True)
        """
        super().__init__(evaluate, initial_param_gen)
        self.simplex = []
        self.w_reflect = w_reflect
        self.w_expand = w_expand
        self.w_contract = w_contract
        self.w_shrink = w_shrink
        self.random_init = random_init

    def init(self):
        vec = np.zeros(self.D)
        vec[0] = 1.
        simplex = [self.initial_param.vector]
        if not self.random_init:
            simplex.extend([np.roll(vec, i) + simplex[0] for i in range(self.D)])  # create simplex
        else:
            simplex.extend([np.random.uniform(size=self.D) + simplex[0] for i in range(self.D)])  # create simplex
        self.simplex = [Param(self.evaluate(param), param) for param in simplex]
        self.simplex.sort(key=lambda x: x.value)

    def iterate(self):
        super().iterate()
        if len(self.simplex) == 0:
            self.init()
        largest  = self.simplex[-1]  # worst performer
        secondlargest = self.simplex[-2]  # second worst performer
        smallest = self.simplex[0]  # best performer
        centroid = np.mean([elem.vector for elem in self.simplex[:-1]], axis=0)

        # reflect in the direction opposite the worst performer
        candidate = normalize(centroid + self.w_reflect*(centroid - largest.vector))
        candidate = Param(self.evaluate(candidate), candidate)

        if candidate.value < smallest.value:
            # that direction was a success, let's try to improve further by expanding
            newcandidate = normalize(centroid + self.w_expand*(candidate.vector - centroid))
            newcandidate = Param(self.evaluate(newcandidate), newcandidate)
            if newcandidate.value < candidate.value:
                logger.debug("operation: expand %.3f" % newcandidate.value)
                self.simplex[-1] = newcandidate
                self.simplex.sort(key=lambda x: x.value)
                return self.simplex[0]

        if candidate.value < secondlargest.value:
            logger.debug("operation: reflect %.3f" % candidate.value)
            self.simplex[-1] = candidate 
            self.simplex.sort(key=lambda x: x.value)
            return self.simplex[0]

        if secondlargest.value <= candidate.value < largest.value:
            # that direction wasn't a complete failure, let's try something more conservative
            logger.debug("operation: outer contract %.3f" % candidate.value)
            newcandidate = normalize(centroid + self.w_contract*(candidate.vector - centroid))
            newcandidate = Param(self.evaluate(newcandidate), newcandidate)
        
        if largest.value <= candidate.value:
            logger.debug("operation: inner contract %.3f" % candidate.value)
            newcandidate = normalize(centroid + self.w_contract*(largest.vector - centroid))
            newcandidate = Param(self.evaluate(newcandidate), newcandidate)

        if newcandidate.value < largest.value:
            self.simplex[-1] = newcandidate
            self.simplex.sort(key=lambda x: x.value)
            return self.simplex[0]

        # form a new simplex closer to the best performer
        logger.debug("operation: shrink all")
        newsimplex = [normalize(smallest.vector + self.w_shrink*(param - smallest.vector)) for param in self.simplex]
        self.simplex = [Param(self.evaluate(param), param) for param in newsimplex]
        self.simplex.sort(key=lambda x: x.value)
        return self.simplex[0]