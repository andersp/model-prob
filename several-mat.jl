using LinearAlgebra
using Plots
using Printf
using Random
using Optim

skew(Z) = 0.5*(Z - Z') # skew-Hermitian part of Z

function init_random_unitary_mat(d::Int64, Vvec::Vector{Matrix{ComplexF64}})
    # make up some random unitary propagators
    nlen = length(Vvec)
    for q=1:nlen
        H0 = rand(ComplexF64,d,d)
        Hsym = 0.5*(H0 + H0')
        Vvec[q] = exp(-im*Hsym)
    end
     
    # final target
    Vtg = copy(Vvec[1])
    for q=2:nlen
        Vtg = Vvec[q]*Vtg
    end
    return Vtg
end

function init_identity_unitary_mat!(d::Int64, Wvec::Vector{Matrix{ComplexF64}})
    nlen = length(Wvec)
    for q=1:nlen
        Wvec[q] = Matrix(I,d,d)
    end
end

function eval_trace_obj(Zvec::Vector{Matrix{ComplexF64}}, Vvec::Vector{Matrix{ComplexF64}}, Vtg::Matrix{ComplexF64})
    # evaluate a sum of trace infidelities
    Nterms = length(Zvec)
    Nwin = length(Vvec) # Vvec has one more element than Zvec
    d = size(Zvec[1],1)

    G = norm(Zvec[1])^2 - abs(tr(Zvec[1]'*Vvec[1]))^2/d # first term
    for q=2:Nterms
        G += norm(Zvec[q])^2 - abs(tr(Zvec[q]'*Vvec[q]*Zvec[q-1]))^2/d
    end
    # last window
    G += norm(Vtg)^2 - abs(tr(Vtg'*Vvec[Nwin]*Zvec[Nwin-1]))^2/d # Final target in Vtg
    return G/Nterms
end

function trace_obj_line(t::Float64, Wvec::Vector{Matrix{ComplexF64}}, Hvec::Vector{Matrix{ComplexF64}}, Vvec::Vector{Matrix{ComplexF64}}, Uvec::Vector{Matrix{ComplexF64}}, Vtg::Matrix{ComplexF64})
    nTerms = length(Wvec) 
    for q=1:nTerms
        Uvec[q] = exp(t * Hvec[q]) * Wvec[q] 
    end
    return eval_trace_obj(Uvec, Vvec, Vtg)
end

function unitary_retraction!(t::Float64, Wvec::Vector{Matrix{ComplexF64}}, Hvec::Vector{Matrix{ComplexF64}}, Uvec::Vector{Matrix{ComplexF64}})
    # evaluate the retraction at parameter 't'
    nTerms = length(Wvec)
    for q=1:nTerms
        Uvec[q] = exp(t * Hvec[q]) * Wvec[q] 
    end
end

function eval_Euclidean_grad!(Zvec::Vector{Matrix{ComplexF64}}, Vvec::Vector{Matrix{ComplexF64}}, Gvec::Vector{Matrix{ComplexF64}}, Vtg::Matrix{ComplexF64})
    # in-place results in Gvec
    Nterms = length(Zvec)
    d = size(Zvec[1],1)

    Gvec[1] = 2*(Zvec[1] - Vvec[1]*tr(Vvec[1]'*Zvec[1])/d)

    for q=2:Nterms
        Gvec[q-1] -= 2*(Vvec[q]'*Zvec[q] *tr(Zvec[q]'*Vvec[q]*Zvec[q-1])/d) # contributions to the previous gradient from Zvec[q]
        Gvec[q] = 2*(Zvec[q] - Vvec[q]*Zvec[q-1]*tr(Zvec[q-1]'*Vvec[q]'*Zvec[q])/d)
    end

    # contribution from last window
    q = Nterms+1 # = number of windows
    Gvec[q-1] -= 2*(Vvec[q]'*Vtg *tr(Vtg'*Vvec[q]*Zvec[q-1])/d)

    # scaling
    Gvec[:] = Gvec./Nterms
end

function inner_prod(Avec::Vector{Matrix{ComplexF64}}, Bvec::Vector{Matrix{ComplexF64}})
    Nterms = length(Avec)
    sum = 0.0
    for q = 1:Nterms
        sum += tr(Avec[q]'*Bvec[q])
    end
    # sum = real(tr(H₁'*N₁) + tr(H₂'*N₂) + tr(H₃'*N₃))
    return real(sum)
end

q = 2 # two qubits
d = 2^q # dimension of matrices (d x d)
Id = Matrix(I,d,d)

# set the seed in the random number generator
Random.seed!(1234)

Nterms = 6 # Number of interior initial conditions in the objective
Nwin = Nterms+1

Vvec = Vector{Matrix{ComplexF64}}(undef,Nwin) # Propagators in each window: Nterms + 1 elements
Wvec = Vector{Matrix{ComplexF64}}(undef,Nterms) # unknown intermediate initial conditions,
Gvec = Vector{Matrix{ComplexF64}}(undef,Nterms) # Euclidean gradient
Svec = Vector{Matrix{ComplexF64}}(undef,Nterms) # Riemannian gradient (current point)
Nvec = Vector{Matrix{ComplexF64}}(undef,Nterms) # Riemannian gradient (next point)
Hvec = Vector{Matrix{ComplexF64}}(undef,Nterms) # Search direction
Uvec = Vector{Matrix{ComplexF64}}(undef,Nterms) # workspace

# Final target
Vtg = init_random_unitary_mat(d, Vvec)
println("Final target dim: ", size(Vtg,1), " norm^2: ", norm(Vtg)^2)

# Initial guess: identity matrices
init_identity_unitary_mat!(d, Wvec)

G0 = eval_trace_obj(Wvec, Vvec, Vtg)
println("Initial objective: ", G0)

# scaling (not used for CG)
t₁_fact = 1.0
t₂_fact = 1.0
t₃_fact = 1.0

# Euclidean gradient
eval_Euclidean_grad!(Wvec, Vvec, Gvec, Vtg)

# Riemannian gradient (these functions are not currently used)
# R₁(W₁,W₂) = skew(∇₁G(W₁,W₂)*W₁') * W₁
# R₂(W₁,W₂,W₃) = skew(∇₂G(W₁,W₂,W₃)*W₂') * W₂
# R₃(W₂,W₃) = skew(∇₃G(W₂,W₃)*W₃') * W₃

# skew-Hermitian matrices from the Euclidian gradient
for q=1:Nterms
    Svec[q] = skew(Gvec[q]*Wvec[q]')
end

# The initial search direction equals S
γ = 0.0
Hvec[:] = Svec[:]

tmax_x = 3.0
tmax_y = 3.0
tmax_z = 3.0
t₁ = LinRange(-tmax_x, tmax_x, 201)
t₂ = LinRange(-tmax_y, tmax_y, 201)
t₃ = LinRange(-tmax_z, tmax_z, 201)

t₀ = 0.0
str_1 = @sprintf("t₁")
str_2 = @sprintf("t₂")
str_3 = @sprintf("t₃")

# ofunc(t₁,t₂) =  trace_obj(U₁(t₁), U₂(t₂), U₃(t₀)) # trace objective function, unitary retraction
# plc_12 = Plots.contourf(t₁_fact*t₁, t₂_fact*t₂, ofunc.(t₁,t₂'), color=:tofino, xaxis=str_1, yaxis=str_2, title="Trace objective, three terms")
# scatter!(plc_12, [0.0], [0.0], lab="Origin")
# println("Cartesian contour plot in variable 'plc_12'")

# ofunc_13(t₁,t₃) =  trace_obj(U₁(t₁), U₂(t₀), U₃(t₃)) # trace objective function, unitary retraction
# plc_13 = Plots.contourf(t₁_fact*t₁, t₃_fact*t₃, ofunc_13.(t₁,t₃'), color=:tofino, xaxis=str_1, yaxis=str_3, title="Trace objective, three terms")
# scatter!(plc_13, [0.0], [0.0], lab="Origin")
# println("Cartesian contour plot in variable 'plc_13'")

# ofunc_23(t₂,t₃) =  trace_obj(U₁(t₀), U₂(t₂), U₃(t₃)) # trace objective function, unitary retraction
# plc_23 = Plots.contourf(t₂_fact*t₂, t₃_fact*t₃, ofunc_23.(t₂,t₃'), color=:tofino, xaxis=str_2, yaxis=str_3, title="Trace objective, three terms")
# scatter!(plc_23, [0.0], [0.0], lab="Origin")
# println("Cartesian contour plot in variable 'plc_23'")

# Complex extrapolation
# ofunc2(t₁,t₂) =  trace_obj(Z₁(t₁), Z₂(t₂), Z₃(t₀)) # trace objective function, Euclidian gradient
# plh = Plots.contourf(t₁_fact*t₁, t₂_fact*t₂, ofunc2.(t₁,t₂'), color=:tofino, xaxis=str_1, yaxis=str_2, title="Trace objective, complex extrapolation")
# scatter!(plh, [0.0], [0.0], lab="Origin", leg=:left)
# println("Contour plot in variable 'plh'")

tmax1 = Nterms # 4.0 # 2.5 # max range for line search
tls = LinRange(0.0, tmax1, 101)

tstr= @sprintf("Opt iterations, t₁_s = %5.2f, t₂_s = %5.2f, t₃_s = %5.2f", t₁_fact, t₂_fact, t₃_fact)
plo = plot(title=tstr, leg=:outerright, size=(800,400), xaxis = "t", yaxis = "G(U₁(-t),U₂(-t))", ylims=(0,4.0))

ofunc3(t) =  trace_obj_line(-t, Wvec, Hvec, Vvec, Uvec) # objective function, fixed direction

tstr = "Trace objective, unitary retraction"

to_init = trace_obj_line(0.0, Wvec, Hvec, Vvec, Uvec)
println("initial guess, objective (G) = ", to_init, " t=0.0")

max_iter = 100
obj_hist = zeros(max_iter)
Niter = 0
# steepest descent with line-search
for iter = 0:max_iter-1
    global obj_hist, γ, Niter

    Niter = iter
    Gkk = inner_prod(Svec, Svec)

    if Gkk < 1e-9
        println("Found local minima after ", Niter, " CG iterations, Gkk = ", Gkk)
        break
    end

    op2 = ofunc3.(tls) # Follow the unitary retraction (for plotting)

    # line search
    result = optimize(ofunc3, -tmax1, tmax1, GoldenSection())

    t_min = Optim.minimizer(result)
    o_min = Optim.minimum(result)
    obj_hist[iter+1] = o_min

    println("iter = ", iter, " Gkk = ", Gkk, " γ = ", γ, ", minimum (G) = ", o_min, " minimizer (t): ", t_min)

    if iter <10
        lstr = "uni-"*string(iter)
        plot!(plo, tls, op2, lab=lstr, lw=3)
        scatter!(plo, [t_min], [o_min], lab=:false)
    end

    # next W:
    unitary_retraction!(-t_min, Wvec, Hvec, Uvec)
    Wvec[:] = Uvec[:]

    eval_Euclidean_grad!(Wvec, Vvec, Gvec, Vtg)
    # skew-Hermitian matrices from the Euclidian gradient
    for q=1:Nterms
        Nvec[q] = skew(Gvec[q]*Wvec[q]')
    end

    Uvec[:] = Nvec[:] - Svec[:]
    γ = inner_prod(Uvec, Nvec)/Gkk

    # next search direction
    Hvec[:] = Nvec[:] + γ .* Hvec[:]

    HGprod = inner_prod(Hvec, Nvec)
    if HGprod < 0.0
        println("HGprod = ", HGprod, " < 0, taking a steepest descent direction")
        Hvec[:] = Nvec[:]
    end

    # save new descent direction
    Svec[:] = Nvec[:]
end
println("Line search plot object stored in variable 'plo'")

tstr = @sprintf("CG conv hist")
plconv = plot(obj_hist[1:Niter], lab=string(Nterms)*"-terms", title=tstr, yaxis=:log10,ylims=(1e-10,1e1), xlabel="Iteration", ylabel="Objective")
println("Convergence plot in variable 'plconv'")

# plot landscape near the local optima

# plf_12 = Plots.contourf(t₁_fact*t₁, t₂_fact*t₂, ofunc.(t₁,t₂'), color=:tofino, xaxis=str_1, yaxis=str_2, title="Trace objective, three terms")
# scatter!(plf_12, [0.0], [0.0], lab="Origin")
# println("Cartesian contour plot in variable 'plf_12'")

# plf_13 = Plots.contourf(t₁_fact*t₁, t₃_fact*t₃, ofunc_13.(t₁,t₃'), color=:tofino, xaxis=str_1, yaxis=str_3, title="Trace objective, three terms")
# scatter!(plf_13, [0.0], [0.0], lab="Origin")
# println("Cartesian contour plot in variable 'plf_13'")

# plf_23 = Plots.contourf(t₂_fact*t₂, t₃_fact*t₃, ofunc_23.(t₂,t₃'), color=:tofino, xaxis=str_2, yaxis=str_3, title="Trace objective, three terms")
# scatter!(plf_23, [0.0], [0.0], lab="Origin")
# println("Cartesian contour plot in variable 'plf_23'")

