import unittest
import numpy as np
from shapes import Shape, Circle

class TestCircle(unittest.TestCase):
    def testDrawCircle(self):
        h = w = 200
        img = np.zeros((h, w, 3))
        radius = 10
        c = Circle(100, 100, (1, 1, 1), 1, radius)

        c.draw(image)
        area = len(np.nonzero(image)[0])

