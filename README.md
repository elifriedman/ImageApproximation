# ImageApproximation
Implementation of some optimization algorithms to approximate an image using simple shapes. The optimization algorithms optimize over the positions, colors, alpha channels, and parameters of the shapes to try to draw an image most similar to the input.

# To run
```
git clone https://github.com/elifriedman/ImageApproximation
cd ImageApproximation
python3 image_approximation.py imgs/mona_lisa.jpg --time_limit 300 --log_interval 10 --optimizer annealing --shape circle
```
or
```
python3 image_approximation.py -h
```
to see all options.

# Project Structure
![System Diagram](https://raw.githubusercontent.com/elifriedman/ImageApproximation/master/imgs/sysdiagram.png)

The [Shape](https://github.com/elifriedman/ImageApproximation/blob/master/shapes.py#L6) class is the base class for creating a new shape that can be used. It is responsible for the parameters of a given shape, and contains the interface for drawing shapes to a PIL image.
The [Optimizer](https://github.com/elifriedman/ImageApproximation/blob/master/optimizers.py#L11) class is the interface for the optimizers. Its subclasses implement the iterate method, which is used to run one iteration of the optimizations.

# Optimization Algorithms
## Nelder Mead Simplex Algorithm
The [Nelder Mead Method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method) is a black box optimization algorithm. If the input parameter vector is a D dimensional vector, then the simplex algorithm will start by creating D points, each orthogonal to the original parameter vector to create a sized D+1 simplex. The algorithm proceeds by evaluating each of the D+1 vectors and heuristically adjusting the points of the simplex to be farther from the worst point and closer to the best point.

For this image approximation problem, this method did not perform well. Since it requires evaluating D+1 points (which in this case is at least 700 (=at least 7 parameters for each shape, 100 shapes)), the algorithm was very slow.

## Particle Swarm Optimization
[Particle Swarm Optimizition](https://en.wikipedia.org/wiki/Particle_swarm_optimization) uses a number of points in parallel to explore the optimization region. Each point travels at a certain randomly chosen velocity, which is updated to take into account the best position that that particle has seen and the best position that any particle has seen.

Finetuning the hyperparameters for PSO is a challenge, and in the end, what worked best were those suggested in the original paper, although results from PSO were underwhelming.

## Simulated Annealing
[Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing) starts with an initial parameter vector and updates it in a random direction, if the random direction improves the evaluation function. However, the algorithm also has a built in exploration mechanism so that it could also accept the random direction with some probability. That probability becomes smaller as time goes on according to the temperature schedule (high temperature -> more randomness, low temperature -> less).

This algorithm worked the best of the three, with some adjustments. Rather than updating the whole parameter vector at once, it would choose one element from the vector to update. This was a more conservative update that kept the new vector closer to the old one and allowed the algorithm to improve previous good results. In addition, the evaluation function needed to change from mean squared error to mean absolute error. Whereas the other algorithms only compare whether one point is less than or greater than another, the probability function in simulated annealing looks at the magnitude of the difference between points, so the smaller difference when using mean absolute error worked better.

# Evaluation Function
Each evaluation compared the mean absolute difference between the candidate image and the target image.

# TODO
- [ ] Improve current algorithms
- [ ] Genetic Algorithm
- [ ] Try out reinforcement learning. Each timestep, propose a new shape and receive a reward = to the evaluation function

# Result
![Mona Lisa](https://raw.githubusercontent.com/elifriedman/ImageApproximation/master/imgs/mona_lisa.jpg)
![Result](https://raw.githubusercontent.com/elifriedman/ImageApproximation/master/imgs/annealing_gamma_0.01_circular_normalization.png)
Result running Simulated Annealing with gamma=0.01 and circular normalization (e.g. 1.2 becomes .2 and -0.2 becomes 0.8) for 15 minutes.
