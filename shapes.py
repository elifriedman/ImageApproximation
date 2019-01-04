from math import sqrt
import numpy as np
from PIL import Image, ImageDraw

class DrawUtils:

    @staticmethod
    def color(image_segment, color, alpha):
        image_segment[:] = alpha*np.array(color) + (1.0 - alpha) * image_segment


class Shape(object):
    """A shape object
    """
    def __init__(self, pos, color, alpha):
        self.x = pos[0]
        self.y = pos[1]
        self.color = color
        self.alpha = alpha
        self._params = [*pos, *color, alpha]


    @classmethod
    def init_from_params(cls, params):
        pos = params[:2]
        color = params[2:5]
        params = params[5:]
        return cls(pos, color, *params)

    @property
    def params(self):
        return self._params

    @classmethod
    def random_params(cls):
        pos = np.random.rand(2)
        color = np.random.rand(3)
        alpha = np.random.rand()*.99  # don't want anything opaque
        return [*pos, *color, alpha]

    def denorm(self, image):
        try:
            h, w, _ = image.shape
        except AttributeError:
            w, h = image.size
        x = int(self.x * h)
        y = int(self.y * w)
        return [(x, y), self.color, self.alpha]

    def draw(self, image):
        self.denorm(image)


class Rectangle(Shape):
    def __init__(self, pos, color, alpha, width, height):
        super().__init__(pos, color, alpha)
        self.width = width
        self.height = height
        self.params.extend([self.width, self.height])

    @classmethod
    def random_params(cls):
        params = super().random_params()
        width, height = np.random.rand(2)
        params.extend([width, height])
        return params

    def denorm(self, image):
        h, w = image.size
        height = int(h*self.height)
        width = int(w*self.width)

        params = super().denorm(image)
        params.extend([width, height])
        return params

    def draw(self, image):
        (x, y), c, a, w, h = self.denorm(image)

        bbox = [(x, y), (x+h, y+w)]
        drw = ImageDraw.Draw(image, 'RGBA')
        drw.rectangle(bbox, tuple([int(c*255) for c in self.color] + [int(self.alpha*255)]))


class Circle(Shape):
    def __init__(self, pos, color, alpha, radius):
        super().__init__(pos, color, alpha)
        self.radius = radius
        self._params.append(self.radius)

    @classmethod
    def random_params(cls):
        params = super().random_params()
        radius = 0.4*np.random.rand()
        params.append(radius)
        return params

    def denorm(self, image):
        params = super().denorm(image)
        params.append(int(self.radius*min(image.size)))
        return params

    def draw(self, image):
        (x, y), color, alpha, radius = self.denorm(image)
        
        topleft = (x - radius, y - radius)
        botright =  (x + radius, y + radius)
        drw = ImageDraw.Draw(image, 'RGBA')
        drw.ellipse((topleft, botright), tuple([int(c*255) for c in self.color] + [int(self.alpha*255)]))


def drawcircs(image, N):
    h = 400
    w = 600
    circles = []
    for i in range(N):
        param = Circle.random_params((h, w))
        c = Circle.init_from_params(param)
        circles.append(c)
        c.draw(image)
    return image, circles