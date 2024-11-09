using LinearAlgebra
using Plots
using Printf
using Random
using Optim

# evaluate a sum of trace infidelities

q = 2 # two qubits
d = 2^q # dimension of matrices (d x d)
Id = Matrix(I,d,d)

# V₁ = get_swap_1d_gate(q) # unitary gate to swap between qubits 1 and 'd'
# V₂ = get_Hd_gate(d) # Take S_2 to be the unitary QFT(d) matrix
# V₃ = get_swap_1d_gate(q)

# construct a random unitary matrices
Random.seed!(1234)

# V1 matrix
H0 = rand(ComplexF64,d,d)
Hsym = 0.5*(H0 + H0')
V₁ = exp(-im*Hsym)

# V2 matrix
H0 = rand(ComplexF64,d,d)
Hsym = 0.5*(H0 + H0')
V₂ = exp(-im*Hsym)

# V3 matrix
H0 = rand(ComplexF64,d,d)
Hsym = 0.5*(H0 + H0')
V₃ = exp(-im*Hsym)

# Initial guess: identity matrices
W₁ = Matrix(I,d,d)
W₂ = Matrix(I,d,d)
W₃ = Matrix(I,d,d)

# scaling (only used in SD)
t₁_fact = 1.0
t₂_fact = 1.0
t₃_fact = 1.0 # 2.0

skew(Z) = 0.5*(Z - Z') # skew-Hermitian part of Z

Nterms = 3 # Number of terms in the objective
trace_obj(Z₁,Z₂,Z₃) = (G₁(Z₁) + G₂(Z₁,Z₂) + G₃(Z₂,Z₃))/Nterms

G₁(Z₁) = norm(Z₁)^2 - abs(tr(Z₁'*V₁))^2/d # Gen infidelity of first part
∇₁G₁(Z₁) = 2*(Z₁ - V₁*tr(V₁'*Z₁)/d ) # Euclidean gradient of G₁

G₂(Z₁,Z₂) = norm(Z₂)^2 - abs(tr(Z₂'*V₂*Z₁))^2/d # only convex when Z₁ is unitary
∇₁G₂(Z₁,Z₂) = 2*(-V₂'*Z₂ *tr(Z₂'*V₂*Z₁)/d)
∇₂G₂(Z₁,Z₂) = 2*(Z₂ - V₂*Z₁*tr(Z₁'*V₂'*Z₂)/d)

∇₁G(Z₁, Z₂) = (∇₁G₁(Z₁) + ∇₁G₂(Z₁,Z₂))/Nterms # Adding up the contributions to the gradient

G₃(Z₂,Z₃) = norm(Z₃)^2 - abs(tr(Z₃'*V₃*Z₂))^2/d # only convex when Z₂ is unitary
∇₂G₃(Z₂,Z₃) = 2*(-V₃'*Z₃ *tr(Z₃'*V₃*Z₂)/d)
∇₃G₃(Z₂,Z₃) = 2*(Z₃ - V₃*Z₂*tr(Z₂'*V₃'*Z₃)/d)

∇₂G(Z₁,Z₂,Z₃) = (∇₂G₂(Z₁,Z₂) + ∇₂G₃(Z₂,Z₃))/Nterms # Adding up the contributions to the gradient

# final component of the gradient only gets one contribution
∇₃G(Z₂,Z₃) = ∇₃G₃(Z₂,Z₃)/Nterms

# Riemannian gradient (these functions are not currently used)
R₁(W₁,W₂) = skew(∇₁G(W₁,W₂)*W₁') * W₁
R₂(W₁,W₂,W₃) = skew(∇₂G(W₁,W₂,W₃)*W₂') * W₂
R₃(W₂,W₃) = skew(∇₃G(W₂,W₃)*W₃') * W₃

# skew-Hermitian retraction matrices:
S₁ = skew(∇₁G(W₁,W₂)*W₁')
S₂ = skew(∇₂G(W₁,W₂,W₃)*W₂')
S₃ = skew(∇₃G(W₂,W₃)*W₃')
    
# The initial search direction equals S
γ = 0.0
H₁ = S₁
H₂ = S₂
H₃ = S₃

# Unitary retractions in the directions (H₁, H₂, H₃)
U₁(t₁) = exp(t₁_fact*t₁ * H₁) * W₁
U₂(t₂) = exp(t₂_fact*t₂ * H₂) * W₂ 
U₃(t₃) = exp(t₃_fact*t₃ * H₃) * W₃ 

# Complex evolution of trace_obj, following the Euclidean gradient
# Z₁(t₁) = W₁ + t₁_fact*t₁ * ∇₁G(W₁,W₂)
# Z₂(t₂) = W₂ + t₂_fact*t₂ * ∇₂G(W₁,W₂,W₃)
# Z₃(t₃) = W₃ + t₃_fact*t₃ * ∇₃G(W₂,W₃)

tmax_x = 3.0/t₁_fact
tmax_y = 3.0/t₂_fact
tmax_z = 3.0/t₃_fact
# tmax_x = 3.0
# tmax_y = 3.0
# tmax_z = 3.0
t₁ = LinRange(-tmax_x, tmax_x, 201)
t₂ = LinRange(-tmax_y, tmax_y, 201)
t₃ = LinRange(-tmax_z, tmax_z, 201)

t₀ = 0.0
str_1 = @sprintf("t₁")
str_2 = @sprintf("t₂")
str_3 = @sprintf("t₃")

ofunc(t₁,t₂) =  trace_obj(U₁(t₁), U₂(t₂), U₃(t₀)) # trace objective function, unitary retraction
plc_12 = Plots.contourf(t₁_fact*t₁, t₂_fact*t₂, ofunc.(t₁,t₂'), color=:tofino, xaxis=str_1, yaxis=str_2, title="Trace objective, three terms")
scatter!(plc_12, [0.0], [0.0], lab="Origin")
println("Cartesian contour plot in variable 'plc_12'")

ofunc_13(t₁,t₃) =  trace_obj(U₁(t₁), U₂(t₀), U₃(t₃)) # trace objective function, unitary retraction
plc_13 = Plots.contourf(t₁_fact*t₁, t₃_fact*t₃, ofunc_13.(t₁,t₃'), color=:tofino, xaxis=str_1, yaxis=str_3, title="Trace objective, three terms")
scatter!(plc_13, [0.0], [0.0], lab="Origin")
println("Cartesian contour plot in variable 'plc_13'")

ofunc_23(t₂,t₃) =  trace_obj(U₁(t₀), U₂(t₂), U₃(t₃)) # trace objective function, unitary retraction
plc_23 = Plots.contourf(t₂_fact*t₂, t₃_fact*t₃, ofunc_23.(t₂,t₃'), color=:tofino, xaxis=str_2, yaxis=str_3, title="Trace objective, three terms")
scatter!(plc_23, [0.0], [0.0], lab="Origin")
println("Cartesian contour plot in variable 'plc_23'")

# Complex extrapolation
# ofunc2(t₁,t₂) =  trace_obj(Z₁(t₁), Z₂(t₂), Z₃(t₀)) # trace objective function, Euclidian gradient
# plh = Plots.contourf(t₁_fact*t₁, t₂_fact*t₂, ofunc2.(t₁,t₂'), color=:tofino, xaxis=str_1, yaxis=str_2, title="Trace objective, complex extrapolation")
# scatter!(plh, [0.0], [0.0], lab="Origin", leg=:left)
# println("Contour plot in variable 'plh'")

tmax1 = 2.5 # max range for line search
tls = LinRange(0.0, tmax1, 101)

tstr= @sprintf("Opt iterations, t₁_s = %5.2f, t₂_s = %5.2f, t₃_s = %5.2f", t₁_fact, t₂_fact, t₃_fact)
plo = plot(title=tstr, leg=:outerright, size=(800,400), xaxis = "t", yaxis = "G(U₁(-t),U₂(-t))", ylims=(0,4.0))

ofunc3(t) =  trace_obj(U₁(-t), U₂(-t), U₃(-t)) # objective function, fixed direction
tstr = "Trace objective, unitary retraction"

to_init = ofunc3(0.0)
println("initial guess, minimum (G) = ", to_init, " t=0.0")

max_iter = 25
obj_hist = zeros(max_iter)
Niter = 0
# steepest descent with line-search, keeping ϕ fixed
for iter = 0:max_iter-1
    global W₁, W₂, W₃, S₁, S₂, S₃, H₁, H₂, H₃, obj_hist, γ, Niter

    Niter = iter
    Gkk = real(tr(S₁'*S₁) + tr(S₂'*S₂) + tr(S₃'*S₃))

    if Gkk < 1e-9
        println("Found local minima after ", Niter, " CG iterations, Gkk = ", Gkk)
        break
    end

    op2 = trace_obj.(U₁.(-tls), U₂.(-tls), U₃.(-tls) ) # Follow the unitary manifold (for plotting)

    # line search (algorithm doesn't make any difference)
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
    Temp1 = U₁(-t_min) # evaluate the final unitaries
    Temp2 = U₂(-t_min)
    Temp3 = U₃(-t_min) 
    W₁ = Temp1
    W₂ = Temp2
    W₃ = Temp3

    # Steepest descent direction at new W
    N₁ = skew(∇₁G(W₁,W₂)*W₁')
    N₂ = skew(∇₂G(W₁,W₂,W₃)*W₂')
    N₃ = skew(∇₃G(W₂,W₃)*W₃')

    # Calculate weight factor
    γ = real(tr((N₁'-S₁')*N₁) + tr((N₂'-S₂')*N₂) + tr((N₃'-S₃')*N₃)) / Gkk

    # next search direction
    H₁ = N₁ + γ * H₁ 
    H₂ = N₂ + γ * H₂
    H₃ = N₃ + γ * H₃

    HGprod = real(tr(H₁'*N₁) + tr(H₂'*N₂) + tr(H₃'*N₃))

    if HGprod < 0.0
        println("HGprod = ", HGprod, " < 0, using steepest descent instead ")
        H₁ = N₁ 
        H₂ = N₂
        H₃ = N₃
    end

    # save steepest descent direction for next iteration
    S₁ = N₁
    S₂ = N₂
    S₃ = N₃
end
println("Line search plot object stored in variable 'plo'")

tstr = @sprintf("CG conv hist, %d matrices", Nterms)
plconv = plot(obj_hist[1:Niter], lab=:none, title=tstr, yaxis=:log10,ylims=(1e-10,1e1), xlabel="Iteration", ylabel="Objective")
println("Convergence plot in variable 'plconv'")

# plot landscape near the local optima

plf_12 = Plots.contourf(t₁_fact*t₁, t₂_fact*t₂, ofunc.(t₁,t₂'), color=:tofino, xaxis=str_1, yaxis=str_2, title="Trace objective, three terms")
scatter!(plf_12, [0.0], [0.0], lab="Origin")
println("Cartesian contour plot in variable 'plf_12'")

plf_13 = Plots.contourf(t₁_fact*t₁, t₃_fact*t₃, ofunc_13.(t₁,t₃'), color=:tofino, xaxis=str_1, yaxis=str_3, title="Trace objective, three terms")
scatter!(plf_13, [0.0], [0.0], lab="Origin")
println("Cartesian contour plot in variable 'plf_13'")

plf_23 = Plots.contourf(t₂_fact*t₂, t₃_fact*t₃, ofunc_23.(t₂,t₃'), color=:tofino, xaxis=str_2, yaxis=str_3, title="Trace objective, three terms")
scatter!(plf_23, [0.0], [0.0], lab="Origin")
println("Cartesian contour plot in variable 'plf_23'")

