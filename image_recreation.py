import time
import numpy as np
from PIL import Image

from shapes import Circle, Rectangle
from optimizers import SimplexOptimizer, SimulatedAnnealing

def vec2params(N, param_vec):
    return np.reshape(param_vec, (N, -1))

def params2vec(params):
    return np.concatenate(params)

class IMG(object):
    def __init__(self, shape):
        self.shape = shape
        self.image = np.zeros(shape)
    
    def get(self):
        self.image.fill(0)
        return self.image

def main():
    N = 100
    target = np.asarray(Image.open('image.jpg'))
    shape_type = Circle

    def genvec():
        params = []
        for i in range(N):
            params.append(shape_type.random_params())
        param_vec = params2vec(params)
        return param_vec

    param_vec = genvec()
    img_gen = IMG(target.shape)
    def evaluate(param_vec):
        params = vec2params(N, param_vec)
        img = Image.new('RGB', (target.shape[1], target.shape[0]))
        for param in params:
            shape = shape_type.init_from_params(param)
            shape.draw(img)
        return np.mean(np.abs(np.asarray(img) - target))
    
    def draw(param_vec):
        params = vec2params(N, param_vec)
        params = np.minimum(1.0, np.maximum(0, params))
        img = Image.new('RGB', (target.shape[1], target.shape[0]))
        for param in params:
            shape = shape_type.init_from_params(param)
            shape.draw(img)
        return img

#    optimizer = SimplexOptimizer
    optimizer = SimulatedAnnealing
    optim = optimizer(evaluate, genvec)
    optim.init()

    start = time.time()
    best_score = 1E9
    for i in range(100000):
        score, param_vec = optim.iterate()
        if score < best_score:
            img = draw(param_vec)
            img.save('best_{}.png'.format(i))
            best_score = score
        if i % 1000 == 0:
            try:
                param_vec = optim.current[1]
            except AttributeError:
                pass
            img = draw(param_vec)
            img.save('img_{}.png'.format(i))
        curtime = time.time() - start
        print("{}: took %.3fs. score=%.3f, op={}".format(i, optim.op) % (curtime, score))
    end = time.time() - start
    print("Total time: {}".format(end))




if __name__ == '__main__':
    main()