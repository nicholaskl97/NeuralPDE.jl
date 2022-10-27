using NeuralPDE
using ModelingToolkit

@parameters x
@variables u(..)

eqs = [u(x) - u(0) ~ 0.]
dict_indvars = Dict(:x => 1)
dict_depvars = Dict(:u => 1)
a1 = NeuralPDE.get_argument(eqs, dict_indvars, dict_depvars)
a2 = NeuralPDE.get_variables(eqs, dict_indvars, dict_depvars)
a3 = NeuralPDE.get_number(eqs, dict_indvars, dict_depvars)
a4 = NeuralPDE.pair(eqs[1], [:u], dict_depvars, dict_indvars)
println(eqs[1])
println("Arguments: ", a1)
println("Variables: ", a2)
println("Numbers: ", a3)
println("Pair: ", a4)
println()

eqs = [u(x) ~ 0.]
b1 = NeuralPDE.get_argument(eqs, dict_indvars, dict_depvars)
b2 = NeuralPDE.get_variables(eqs, dict_indvars, dict_depvars)
b3 = NeuralPDE.get_number(eqs, dict_indvars, dict_depvars)
println(eqs[1])
println("Arguments: ", b1)
println("Variables: ", b2)
println("Numbers: ", b3)
println()

eqs = [u(0) ~ 0.]
c1 = NeuralPDE.get_argument(eqs, dict_indvars, dict_depvars)
c2 = NeuralPDE.get_variables(eqs, dict_indvars, dict_depvars)
c3 = NeuralPDE.get_number(eqs, dict_indvars, dict_depvars)
println(eqs[1])
println("Arguments: ", c1)
println("Variables: ", c2)
println("Numbers: ", c3)
println()