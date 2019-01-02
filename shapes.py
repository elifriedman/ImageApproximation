from math import sqrt
import numpy as np

class DrawUtils:

    @staticmethod
    def color(image_segment, color, alpha):
        image_segment[:] = alpha*color + (1.0 - alpha) * image_segment

class Shape(object):
    """A shape object
    """

    def __init__(self, x, y, color, alpha):
        self.x = x
        self.y = y
        self.color = np.asarray(color)
        self.alpha = alpha
        self._params = [self.x, self.y, self.color, alpha]


    @classmethod
    def init_from_params(cls, params):
        return cls(*params)

    @property
    def params(self):
        return np.asarray(self._params)

    def draw(self, image):
        pass

class Circle(Shape):
    def __init__(self, x, y, color, alpha, radius):
        super().__init__(x, y, color, alpha)
        self.radius = radius
        self._params.append(self.radius)

    def draw(self, image):
        rx = self.radius
        rx2 = rx**2
        for i in range(0, self.radius):
            topx = min(image.shape[0]-1, max(self.x + i, 0))
            botx = min(image.shape[0]-1, max(self.x - i, 0))

            left  = min(image.shape[1]-1, max(-rx + self.y, 0))
            right = min(image.shape[1]-1, max(rx + self.y, 0))
            DrawUtils.color(image[topx, left:right], self.color, self.alpha)
            if i > 0:
                DrawUtils.color(image[botx, left:right], self.color, self.alpha)
            rx2 = rx2 - 2*i - 1
            rx = int(sqrt(rx2 if rx2 >= 0 else 0))

def drawcircs(image, N):
    h = 400
    w = 600
    for i in range(N):
        x = np.random.randint(0, h)
        y = np.random.randint(0, w)
        color = np.random.rand(3)
        alpha = np.random.rand()
        radius = np.random.randint(0, max(h, w)//2)
        param = [x, y, color, alpha, radius]
        c = Circle.init_from_params(param)
        c.draw(image)
    return image
    