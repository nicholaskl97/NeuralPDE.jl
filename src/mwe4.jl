ex = :(u(x,0,y-5))
indvars = ex.args[2:end]

var_ = :($(Expr(:$, :u)))

ex2 = :()
ex2.head = ex.head
#ex2.args = [var_, :( $indvars )] 
ex2.args = [var_, :( [$(indvars...)] )]