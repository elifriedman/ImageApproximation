from math import sqrt
import numpy as np
from PIL import Image, ImageDraw


class Shape(object):
    """A shape object. Takes care of all the shape's parameters.
    """
    def __init__(self, pos, color, alpha):
        """A shape object. Takes care of all the shape's parameters.

        Parameters
        ----------
        pos : tuple (float, float)
            x, y position of the shape. Clipped to be in [0, 1]
        color : tuple (float, float, float)
            r, g, b color of the shape. Clipped to be in [0, 1]
        alpha : float
            opacity of the shape. Clipped to be in range [0, 1]
        """
        self.x = max(0., min(1., pos[0]))
        self.y = max(0., min(1., pos[1]))
        self.color = np.maximum(0., np.minimum(1., color))
        self.alpha = max(0., min(1., alpha))
        self._params = [*pos, *color, alpha]


    @classmethod
    def init_from_params(cls, params):
        """Given a list of shape parameters, instantiate a shape

        Parameters
        ----------
        params : list (float)
            parameters to use to instantiate shape
        """
        assert len(params) > 6, "params must contain position, color, alpha"
        pos = params[:2]
        color = params[2:5]
        params = params[5:]
        return cls(pos, color, *params)

    @property
    def params(self):
        return self._params

    @classmethod
    def random_params(cls):
        """Generate random parameters for this shape
        """
        pos = np.random.rand(2)
        color = np.random.rand(3)
        alpha = np.random.rand()*.99  # don't want anything opaque
        return [*pos, *color, alpha]

    def denorm(self, image):
        """Denormalize parameters so they can be used with an image
        """
        try:
            h, w, _ = image.shape  # numpy 
        except AttributeError:
            w, h = image.size  # PIL
        x = int(self.x * h)
        y = int(self.y * w)
        return [(x, y), self.color, self.alpha]

    def draw(self, image):
        """Draw shape on an image

        Parameters
        ----------
        image : PIL image
        """
        self.denorm(image)


class Rectangle(Shape):
    def __init__(self, pos, color, alpha, width, height):
        """A rectangle shape
        
        Parameters
        ----------
        width : float in [0, 1]
        height : float in [0, 1]
        """
        super().__init__(pos, color, alpha)
        self.width = max(0., min(1., width))
        self.height = max(0., min(1., height))
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
        """A circle shape
        
        Parameters
        ----------
        radius : float in [0, 1]
        """
        super().__init__(pos, color, alpha)
        self.radius = max(0., min(1., radius))
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

class Triangle(Shape):
    def __init__(self, pos, color, alpha, x2, y2, x3, y3):
        """A circle shape
        
        Parameters
        ----------
        x2 : float in [0, 1]
        y2 : float in [0, 1]
        x3 : float in [0, 1]
        y3 : float in [0, 1]
        """
        super().__init__(pos, color, alpha)
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3
        self._params.extend([x2, y2, x3, y3])

    @classmethod
    def random_params(cls):
        params = super().random_params()
        params.extend(np.random.rand(4))
        return params

    def denorm(self, image):
        params = super().denorm(image)
        try:
            h, w, _ = image.shape  # numpy 
        except AttributeError:
            w, h = image.size  # PIL

        x2 = int(self.x2*h)
        x3 = int(self.x2*h)
        y2 = int(self.x3*w)
        y3 = int(self.y3*w)
        params.extend([x2, y2, x3, y3])
        return params

    def draw(self, image):
        (x, y), color, alpha, x2, y2, x3, y3 = self.denorm(image)
        
        drw = ImageDraw.Draw(image, 'RGBA')
        drw.polygon([x, y, x2, y2, x3, y3], tuple([int(c*255) for c in self.color] + [int(self.alpha*255)]))



def drawshapes(shape, image, N):
    shapes = []
    for i in range(N):
        param = Circle.random_params()
        c = Circle.init_from_params(param)
        shapes.append(c)
        c.draw(image)
    return image, shapes