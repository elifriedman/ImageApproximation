import time
import signal
import logging
import numpy as np
import argparse
from PIL import Image

from shapes import Circle, Rectangle, Triangle
from optimizers import SimplexOptimizer, SimulatedAnnealing, ParticleSwarm



def get_shape(shape_name):    
    shape_name = shape_name.lower()
    if shape_name == "circle":
        return Circle
    if shape_name == "rectangle":
        return Rectangle
    if shape_name == "triangle":
        return Triangle
    raise AttributeError("No shape found '{}'".format(shape_name))


class ImageApproximator(object):
    def __init__(self, image_name, shape, num_shapes, optimizer_type, loglevel=logging.INFO):
        self.target = np.asarray(Image.open(image_name))
        self.num_shapes = num_shapes
        self.shape_type = shape
        self.optimizer = optimizer_type(self.evaluate, self.gen_random_vec)
        self.config_logging(loglevel)

    def config_logging(self, loglevel):
        self.logger = logging.getLogger("ImageApprixmator")
        handler = logging.StreamHandler()
        self.logger.addHandler(handler)
        self.logger.setLevel(loglevel)
        handler.setLevel(loglevel)
        logging.getLogger("optimizer").setLevel(loglevel)
        logging.getLogger("optimizer").addHandler(handler)

    def gen_random_vec(self):
        params = []
        for i in range(self.num_shapes):
            params.append(self.shape_type.random_params())
        param_vec = self.params2vec(params)
        return param_vec

    def vec2params(self, param_vec):
        return np.reshape(param_vec, (self.num_shapes, -1))

    def params2vec(self, params):
        return np.concatenate(params)

    def evaluate(self, param_vec, return_img=False):
        params = self.vec2params(param_vec)
        img = Image.new('RGB', (self.target.shape[1], self.target.shape[0]))
        for param in params:
            shape = self.shape_type.init_from_params(param)
            shape.draw(img)
        value = np.mean(np.abs(np.asarray(img) - self.target))
        if return_img:
            return value, img
        return value

    def run(self, num_steps=None, time_limit=None, output_file="output.png", log_interval=1000):
        num_steps = 1000000000 if num_steps is None else num_steps 
        time_limit = 3600*24 if time_limit is None else time_limit

        self.stop = False
        def sig_handler(sig, frame):
            self.stop = True
        signal.signal(signal.SIGINT, sig_handler)

        best = np.inf
        start = time.time()
        for i in range(num_steps):
            now = time.time() - start
            if now >= time_limit:
                break
            if self.stop:
                break

            value, paramvec = self.optimizer.iterate()
            if value < best:
                best = value
                value, img = self.evaluate(paramvec, return_img=True)
                img.save(output_file)
            if i % log_interval == 0:
                now = time.time() - start
                self.logger.info("%d: time=%.3f, score=%.3f", i, now, value)

        return self.evaluate(paramvec, return_img=True)


def get_optimizer(args):
    if args.optimizer == "simplex":
        opt = SimplexOptimizer
    if args.optimizer == "annealing":
        opt = SimulatedAnnealing
    if args.optimizer == "pso":
        opt = ParticleSwarm
    return lambda evl, gen: opt(evl, gen, **args.__dict__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_name", type=str)
    parser.add_argument("--output", type=str, default="output.png", help="output file name")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_interval", type=int, default=1000, help="how often to log")
    parser.add_argument("--num_steps", type=int, help="how many steps to run for")
    parser.add_argument("--time_limit", type=int, help="how long to run for (seconds)")
    parser.add_argument("--shape", type=str, help="shape to use for image approximation",
                                    default="circle",
                                    choices=["circle", "rectangle", "triangle"])
    parser.add_argument("--num_shapes", type=int, default=100, help="number of shapes to use")
    parser.add_argument("--optimizer", type=str,
                                     help="Choose which optimizer to use.",
                                     default="annealing",
                                     choices=["simplex",
                                              "annealing",
                                              "pso"])

    group = parser.add_argument_group("simplex", "Nelson-Mead Simplex Algorithm")
    group.add_argument("--random_init", action="store_true", default=False,
                       help="initialize in random directions as opposed to along axes")
    group.add_argument("--w_reflect", type=float, default=1., help="weight for reflecting")
    group.add_argument("--w_expand", type=float, default=1., help="weight for expanding")
    group.add_argument("--w_contract", type=float, default=1., help="weight for contracting")
    group.add_argument("--w_shrink", type=float, default=1., help="weight for shrinking")

    group = parser.add_argument_group("annealing", "Simulated Annealing")
    group.add_argument("--w_gamma", type=float, default=.1, help="initial multiplier for temperature")
    group.add_argument("--num_axes", type=int, default=1, help="The number of axes of the current point that should be updated per iteration")

    group = parser.add_argument_group("pso", "Particle Swarm Optimization")
    group.add_argument("--number_particles", type=int, default=40, help="Number of particles")
    group.add_argument("--w_omega", type=float, default=0.5 / np.log(2), help="Weight to multipliy the current velociy")
    group.add_argument("--w_pb", type=float, default=0.5 + np.log(2), help="Weight to move in the direction of the personal best")
    group.add_argument("--w_gb", type=float, default=0.5 + np.log(2), help="Weight to move in the direction of the global best")
    group.add_argument("--randomize_order", action="store_true", default=True, help="Randomize order of the particle updates")

    return parser.parse_args()

def main():
    args = parse_args()

    Shape = get_shape(args.shape)
    Optimizer = get_optimizer(args)
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    runner = ImageApproximator(args.image_name, Shape, args.num_shapes, Optimizer, loglevel=loglevel)
    runner.run(args.num_steps, args.time_limit, args.output, args.log_interval)

if __name__ == '__main__':
    main()