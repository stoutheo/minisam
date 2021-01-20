#!/usr/bin/env python3

# Following the example described at : https://minisam.readthedocs.io/pose_graph_2d.html

import minisam as ms
import minisam.sophus as msphu
from utils import plotSE2WithCov

import numpy as np
import math
import matplotlib.pyplot as plt


# Create the state variable
stVariable1 = ms.key('x', 1)
stVariable2 = ms.key('x', 2)
stVariable3 = ms.key('x', 3)
stVariable4 = ms.key('x', 4)
stVariable5 = ms.key('x', 5)


# ------------------------------------------------------------------------------

# initial variable values --- current best estimate
# add random noise from ground truth values
initials = ms.Variables()
initials.add(stVariable1, msphu.SE2(msphu.SO2(0.2), np.array([1.2, -0.3])))
initials.add(stVariable2, msphu.SE2(msphu.SO2(-0.1), np.array([5.1, 0.3])))
initials.add(stVariable3, msphu.SE2(msphu.SO2(-1.57 - 0.2), np.array([9.9, -0.1])))
initials.add(stVariable4, msphu.SE2(msphu.SO2(-3.14 + 0.1), np.array([10.2, -5.0])))
initials.add(stVariable5, msphu.SE2(msphu.SO2(1.57 - 0.1), np.array([5.1, -5.1])))

# ------------------------------------------------------------------------------


# factor graph container
graph = ms.FactorGraph()

# ------------------------------------------------------------------------------
# Factors represent the measurements
# ------------------------------------------------------------------------------


# Add a prior on the first pose, setting it to the origin

# Create a diagonal covariance 3x3 matrix
# either via Sigmas
priorLoss = ms.DiagonalLoss.Sigmas(np.array([2.0, 2.0, 0.1])) # prior loss function
# or via the Precision = 1/(sigmas^2)
# priorLoss = ms.DiagonalLoss.Precisions(np.array([1.0, 1.0, 100.])) # prior loss function

# it prints the 1/sigma
# print(priorLoss)

# Create a factor that represents the prior : PriorFactor

# define the pose
# create a 2 dimensional group of rigid body transformations out of a:
    # 2-dim special orthogonal group matrices ---> 2D rotation matrix (orientation)
    # 2-dim vector  ---> 2D translation (position)
pose = msphu.SE2(msphu.SO2(0), np.array([0, 0])) # ----- These values represent the measurement

# Create the factor with name, pose, and uncertainty
priorFct = ms.PriorFactor(stVariable1, pose, priorLoss)

# add the prior factor to the graph
graph.add(priorFct)

print("Added the prior factor: ")
print(graph, "\n")

# ------------------------------------------------------------------------------
# Create a factor that represents the odomentry : BetweenFactor

# Create a diagonal covariance 3x3 matrix
odomLoss = ms.DiagonalLoss.Sigmas(np.array([0.5, 0.5, 0.1])) # odometry measurement loss function or variance

# Add odometry factors

# Create a factor that represents the odomentry : BetweenFactor
# Create the factor with two state variables, a 2D transformation, and uncertainty
# move 5 on local x axis
graph.add(ms.BetweenFactor(stVariable1, stVariable2, msphu.SE2(msphu.SO2(0), np.array([5, 0])), odomLoss))
# move 5 on local x axis and rotate 90 deg
graph.add(ms.BetweenFactor(stVariable2, stVariable3, msphu.SE2(msphu.SO2(-1.57), np.array([5, 0])), odomLoss))
# move 5 on local x axis and rotate 90 deg
graph.add(ms.BetweenFactor(stVariable3, stVariable4, msphu.SE2(msphu.SO2(-1.57), np.array([5, 0])), odomLoss))
# move 5 on local x axis and rotate 90 deg
graph.add(ms.BetweenFactor(stVariable4, stVariable5, msphu.SE2(msphu.SO2(-1.57), np.array([5, 0])), odomLoss))

print("Added the 4 odometry factors: ")
print(graph, "\n")

# ------------------------------------------------------------------------------
# Create a factor that represents the loop closure : BetweenFactor

# Create a diagonal covariance 3x3 matrix
loopLoss = ms.DiagonalLoss.Sigmas(np.array([0.5, 0.5, 0.1])) # loop closure measurement loss function

# Add the loop closure constraint
# Create a factor that represents the  loop closure  : BetweenFactor
# Create the factor with two state variables, a 2D transformation, and uncertainty
# So now we have except the Markov chain relation and arbitary geometric relation between two factors x5 <--> x2
graph.add(ms.BetweenFactor(stVariable5, stVariable2, msphu.SE2(msphu.SO2(-1.57), np.array([5, 0])), loopLoss))

print("Added the loop closure factor: ")
print(graph, "\n")


# ------------------------------------------------------------------------------
# Solve the mean estimation problem via non-linear least squares
# ------------------------------------------------------------------------------

#  LM method optimizes the initial values
opt_param = ms.LevenbergMarquardtOptimizerParams()
opt = ms.LevenbergMarquardtOptimizer(opt_param)

# result variables container
results = ms.Variables()
status = opt.optimize(graph, initials, results)

if status != ms.NonlinearOptimizationStatus.SUCCESS:
    print("optimization error: ", status)
else:
    print("optimization success")

print("Estimated values for the variables are: ")
print(results)


# ------------------------------------------------------------------------------
# Solve the covariance estimation problem via non-linear least squares
# ------------------------------------------------------------------------------

# Calculate marginal covariances for poses
mcov_solver = ms.MarginalCovarianceSolver()

status = mcov_solver.initialize(graph, results)
if status != ms.MarginalCovarianceSolverStatus.SUCCESS:
    print("maginal covariance error", status)

cov1 = mcov_solver.marginalCovariance(stVariable1)
cov2 = mcov_solver.marginalCovariance(stVariable2)
cov3 = mcov_solver.marginalCovariance(stVariable3)
cov4 = mcov_solver.marginalCovariance(stVariable4)
cov5 = mcov_solver.marginalCovariance(stVariable5)


# ------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------

# plot
fig, ax = plt.subplots()

plotSE2WithCov(results.at(stVariable1), cov1)
plotSE2WithCov(results.at(stVariable2), cov2)
plotSE2WithCov(results.at(stVariable3), cov3)
plotSE2WithCov(results.at(stVariable4), cov4)
plotSE2WithCov(results.at(stVariable5), cov5)

plt.axis('equal')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.show()





# end
