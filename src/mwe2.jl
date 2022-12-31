using Flux, NeuralPDE, Test
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using QuasiMonteCarlo
import ModelingToolkit: Interval, infimum, supremum
using DomainSets
using Random
using CUDA
import Lux

Random.seed!(110)

println("Approximation of function 1D")

@parameters x
@variables u(..)
func(x) = @. cos(pi * x) * x
eq = [u(x) - u(0) ~ func(x)]
#eq = [u(x) ~ func(x)]
bc = [u(0) ~ u(0)]

x0 = 0
x_end = 0.5
domain = [x ∈ Interval(x0, x_end)]

hidden = 20
chain = Lux.Chain(Lux.Dense(1, hidden, Lux.sin),
                  Lux.Dense(hidden, hidden, Lux.sin),
                  Lux.Dense(hidden, hidden, Lux.sin),
                  Lux.Dense(hidden, 1, bias=false))

# Train on GPU
ps = Lux.setup(Random.default_rng(), chain)[1]
ps = ps |> Lux.ComponentArray |> gpu .|> Float64

#strategy = NeuralPDE.GridTraining(0.01)
strategy = NeuralPDE.QuasiRandomTraining(10000;bcs_points=0)

discretization = NeuralPDE.PhysicsInformedNN(chain, strategy, init_params = ps)
@named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
prob = NeuralPDE.discretize(pde_system, discretization)

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 500)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(), maxiters = 1000)

dx = 0.01
xs = collect(x0:dx:x_end)
func_s = func(xs)
# func_approx(x) = discretization.phi(x, res.u) .- discretization.phi(0., res.u)
func_approx(x) = first.(vec(Array(discretization.phi(gpu(x), res.u)) .- Array(discretization.phi(gpu(0.), res.u))))


using Plots

plot(xs, func(xs')', label="True function")
plot!(xs, func_approx(xs'), label="Approximate function")

@test func_approx(xs')≈func(xs') rtol=0.01