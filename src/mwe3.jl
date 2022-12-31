using NeuralPDE
import ModelingToolkit: Interval, infimum, supremum
using DomainSets

# Set up problem

@parameters x
@variables u(..)

func(x) = @. cos(5pi * x) * x
eqs = [u(x) - u(0) ~ func(x)]
bcs = [u(0) ~ u(0)]

x0 = 0
x_end = 4
domains = [x ∈ Interval(x0, x_end)]

# Set up pinnrep

eq_params = SciMLBase.NullParameters()
defaults = Dict{Any, Any}()
default_p = nothing
param_estim = false
additional_loss = nothing
adaloss = nothing
depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = NeuralPDE.get_vars([x],[u(x)])
logger = nothing
multioutput = false
iteration = [1]
init_params = nothing
flat_init_params = nothing
phi = nothing
derivative = nothing
strategy = NeuralPDE.GridTraining(0.01)

pde_indvars = if true# strategy isa QuadratureTraining
    NeuralPDE.get_argument(eqs, dict_indvars, dict_depvars)
else
    NeuralPDE.get_variables(eqs, dict_indvars, dict_depvars)
end

bc_indvars = if strategy isa QuadratureTraining
    NeuralPDE.get_argument(bcs, dict_indvars, dict_depvars)
else
    NeuralPDE.get_variables(bcs, dict_indvars, dict_depvars)
end

pde_integration_vars = NeuralPDE.get_integration_variables(eqs, dict_indvars, dict_depvars)
bc_integration_vars = NeuralPDE.get_integration_variables(bcs, dict_indvars, dict_depvars)

pinnrep = NeuralPDE.PINNRepresentation(eqs, bcs, domains, eq_params, defaults, default_p,
                                 param_estim, additional_loss, adaloss, depvars, indvars,
                                 dict_indvars, dict_depvars, dict_depvar_input, logger,
                                 multioutput, iteration, init_params, flat_init_params, phi,
                                 derivative,
                                 strategy, pde_indvars, bc_indvars, pde_integration_vars,
                                 bc_integration_vars, nothing, nothing, nothing, nothing)

# Attemp to build loss function expressions

eq = first(pinnrep.eqs)

eq_rhs = isequal(expand_derivatives(eq.rhs), 0) ? eq.rhs : expand_derivatives(eq.rhs)
eq_rhs_expr = toexpr(eq_rhs)
right_expr = NeuralPDE._transform_expression(pinnrep, eq_rhs_expr)

eq_lhs = isequal(expand_derivatives(eq.lhs), 0) ? eq.lhs : expand_derivatives(eq.lhs)
eq_lhs_expr = toexpr(eq_lhs)
left_expr = NeuralPDE._transform_expression(pinnrep, eq_lhs_expr)

println("left_expr = ", left_expr)

left_expr_dot = NeuralPDE._dot_(left_expr)

println("_dot_(left_expr) = ", left_expr_dot)

println("parse_equation(", eq,") = ", NeuralPDE.parse_equation(pinnrep, eq))

#a = Expr(:$, :x)
#b = :((vcat)(Any[$a]))
#println(NeuralPDE._dot_(b))
#c = :( $(Expr(:$, :u))($b) )
#println(NeuralPDE._dot_(c))
