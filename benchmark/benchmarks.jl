using BenchmarkTools
using Distributions
using Dynare
using KalmanFilterTools
using LinearAlgebra

SUITE = BenchmarkGroup()

#=
SUITE["lik-small"] = BenchmarkGroup(["likelihood", "small"])

ny = 3
ns = 10
np   = 2
nobs = 50

y = randn(ny, nobs)
T = randn(ns, ns)
Tv = eigen(T).values
T = (1/(maximum(abs.(Tv))+0.0001))*T
R = randn(ns, np)
Q = randn(np, np)
Q = transpose(Q)*Q
Z = randn(ny, ns)
H = randn(ny, ny)
H = transpose(H)*H
s_0 = randn(ns)
s = similar(s_0)
P_0 = randn(ns, ns)
P_0 = transpose(P_0)*P_0
P = similar(P_0)

SUITE["lik-small"]["ws"] = @benchmarkable KalmanLikelihoodWs{Float64, Int64}($ny, $ns, $np, $nobs)
ws1 = KalmanLikelihoodWs{Float64, Int64}(ny, ns, np, nobs)

copy!(s, s_0)
copy!(P, P_0)
SUITE["lik-small"]["lik"] = @benchmarkable kalman_likelihood($y, $Z, $H, $T, $R, $Q, $s, $P, 1, $nobs, 0, $ws1)
=#

context = @dynare "test/models/example5/example5_est_d"
datafile = "fsdata_simul.csv";
dimension(ld::Dynare.DSGELogPosteriorDensity) = ld.dimension;
logdensity(ld::Dynare.DSGELogPosteriorDensity, x) = ld.f(x);
problem = Dynare.DSGELogPosteriorDensity(context, datafile, 1, 0);
(p0, v0) = Dynare.get_initial_values(context.work.estimated_parameters);


SUITE["estimation"] = BenchmarkGroup(["estimation", "likelihood"])
SUITE["estimation"]["likelihood"] = @benchmarkable problem.f(p0)
