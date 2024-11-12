using LinearAlgebra
using Plots
using Printf
using Random
using Optim

skew(Z) = 0.5*(Z - Z') # skew-Hermitian part of Z

function init_random_unitary_mat!(d::Int64, Vvec::Vector{Matrix{ComplexF64}})
    nlen = length(Vvec)
    for q=1:nlen
        H0 = rand(ComplexF64,d,d)
        Hsym = 0.5*(H0 + H0')
        Vvec[q] = exp(-im*Hsym)
    end
end

function init_Vtarget(d::Int64, Vvec::Vector{Matrix{ComplexF64}})
    nlen = length(Vvec)
    # Random solution operator in the last window
    H0last = rand(ComplexF64,d,d)
    Vlast = exp(-im*0.5*(H0last + H0last'))

    # Reachable final-time target: Vtar = Vlast * prod_q Vvec[q]
    Vtar = Vvec[1]
    for q=2:nlen
        Vtar = Vvec[q]*Vtar
    end
    Vtar = Vlast*Vtar

    # # Non-reachable (random) final-time target
    # H0tar = rand(ComplexF64,d,d)
    # Vtar = exp(-im*0.5*(H0tar + H0tar'))

    return Vlast, Vtar
end

function init_identity_unitary_mat!(d::Int64, Wvec::Vector{Matrix{ComplexF64}})
    nlen = length(Wvec)
    for q=1:nlen
        Wvec[q] = Matrix(I,d,d)
    end
end

function eval_trace_per_window(Zvec::Vector{Matrix{ComplexF64}}, Vvec::Vector{Matrix{ComplexF64}})
    nlen = length(Zvec)
    d = size(Zvec[1],1)

    G = zeros(nlen)
    G[1] = norm(Zvec[1])^2 - abs(tr(Zvec[1]'*Vvec[1]))^2/d 
    for q=2:nlen
        G[q] = norm(Zvec[q])^2 - abs(tr(Zvec[q]'*Vvec[q]*Zvec[q-1]))^2/d
    end
    return G
end


function eval_trace_obj(Zvec::Vector{Matrix{ComplexF64}}, Vvec::Vector{Matrix{ComplexF64}}, Vtar::Matrix{ComplexF64}, Vlast::Matrix{ComplexF64})
    # evaluate a sum of trace infidelities
    nlen = length(Zvec)
    d = size(Zvec[1],1)

    # Intermediate window residuals 
    G = sum(eval_trace_per_window(Zvec, Vvec)) / nlen
    # Final time target residual
    G += norm(Vtar)^2 - abs(tr(Vtar'*Vlast*Zvec[nlen]))^2/d
    return G
end

function trace_obj_line(t::Float64, Wvec::Vector{Matrix{ComplexF64}}, Hvec::Vector{Matrix{ComplexF64}}, Vvec::Vector{Matrix{ComplexF64}}, Uvec::Vector{Matrix{ComplexF64}}, Vtar::Matrix{ComplexF64}, Vlast::Matrix{ComplexF64})
    nTerms = length(Wvec) 
    for q=1:nTerms
        Uvec[q] = exp(t * Hvec[q]) * Wvec[q] 
    end
    return eval_trace_obj(Uvec, Vvec, Vtar, Vlast)
end

function unitary_retraction!(t::Float64, Wvec::Vector{Matrix{ComplexF64}}, Hvec::Vector{Matrix{ComplexF64}}, Uvec::Vector{Matrix{ComplexF64}})
    # evaluate the retraction at parameter 't'
    nTerms = length(Wvec)
    for q=1:nTerms
        Uvec[q] = exp(t * Hvec[q]) * Wvec[q] 
    end
end

function eval_Euclidean_grad!(Zvec::Vector{Matrix{ComplexF64}}, Vvec::Vector{Matrix{ComplexF64}}, Vtar::Matrix{ComplexF64}, Vlast::Matrix{ComplexF64}, Gvec::Vector{Matrix{ComplexF64}})
    # in-place results in Gvec
    Nterms = length(Zvec)
    d = size(Zvec[1],1)

    Gvec[1] = 2*(Zvec[1] - Vvec[1]*tr(Vvec[1]'*Zvec[1])/d) / Nterms

    for q=2:Nterms
        Gvec[q-1] -= 2*(Vvec[q]'*Zvec[q] *tr(Zvec[q]'*Vvec[q]*Zvec[q-1])/d) / Nterms # contributions to the previous gradient from Zvec[q]
        Gvec[q] = 2*(Zvec[q] - Vvec[q]*Zvec[q-1]*tr(Zvec[q-1]'*Vvec[q]'*Zvec[q])/d) / Nterms
    end

    Gvec[Nterms] -= 2*(Vlast'*Vtar*tr(Vtar'*Vlast*Zvec[Nterms])/d)

    Gvec[:] = Gvec
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

# Optimize with CG for given number of windows (Nterms) and maximum linesearch 'tmax'
function runCG(Nterms, tmax1)
    # set the seed in the random number generator
    Random.seed!(1234)

    Vvec = Vector{Matrix{ComplexF64}}(undef,Nterms)
    Wvec = Vector{Matrix{ComplexF64}}(undef,Nterms)
    Gvec = Vector{Matrix{ComplexF64}}(undef,Nterms)
    Svec = Vector{Matrix{ComplexF64}}(undef,Nterms)
    Nvec = Vector{Matrix{ComplexF64}}(undef,Nterms)
    Hvec = Vector{Matrix{ComplexF64}}(undef,Nterms)
    Uvec = Vector{Matrix{ComplexF64}}(undef,Nterms) # workspace

    # Set random solution operators for each window
    init_random_unitary_mat!(d, Vvec)

    # Set the tinal-time target unitary and solution operator. Vtar = Vlast*prod_q Vvec[q]
    Vlast, Vtar = init_Vtarget(d, Vvec)

    # Initial guess: identity matrices
    init_identity_unitary_mat!(d, Wvec)

    # Euclidean gradient
    eval_Euclidean_grad!(Wvec, Vvec, Vtar, Vlast, Gvec)

    # skew-Hermitian matrices from the Euclidian gradient
    for q=1:Nterms
        Svec[q] = skew(Gvec[q]*Wvec[q]')
    end

    # The initial search direction equals S
    global γ = 0.0
    Hvec[:] = Svec[:]

    # Initial objective funciton value
    to_init = trace_obj_line(0.0, Wvec, Hvec, Vvec, Uvec, Vtar, Vlast)
    println("initial guess, objective (G) = ", to_init, " t=0.0")

    # Define line search objective function with fixed direction
    ofunc3(t) =  trace_obj_line(-t, Wvec, Hvec, Vvec, Uvec, Vtar, Vlast) 

    max_iter = 500
    global obj_hist = zeros(max_iter)
    global Gkk_hist = zeros(max_iter)
    global residuals = zeros(Nterms, max_iter)
    global Niter = 0

    # CG iterations with line-search
    for iter = 0:max_iter-1

        Niter = iter
        Gkk = inner_prod(Svec, Svec)
        Gkk_hist[iter+1] = Gkk

        if Gkk < 1e-9
            println("Found local minima after ", Niter, " CG iterations, Gkk = ", Gkk)
            break
        end

        # line search
        result = optimize(ofunc3, -tmax1, tmax1, GoldenSection())

        t_min = Optim.minimizer(result)
        o_min = Optim.minimum(result)
        obj_hist[iter+1] = o_min

        println("iter = ", iter, " Gkk = ", Gkk, " γ = ", γ, ", minimum (G) = ", o_min, " minimizer (t): ", t_min)

        # next W:
        unitary_retraction!(-t_min, Wvec, Hvec, Uvec)
        Wvec[:] = Uvec[:]

        # Store residuals per window
        residuals[:, iter+1]  = eval_trace_per_window(Wvec, Vvec)

        eval_Euclidean_grad!(Wvec, Vvec, Vtar, Vlast, Gvec)
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

    return obj_hist[1:Niter], Gkk_hist[1:Niter], residuals[:,1:Niter]
end # of function runCG


##########
# Run CG on increasing numbers of windows
##########

# Number of windows in the objective
# Nterms_all = [2,3,4,5,6]
Nterms_all = [2,4,8, 16, 32]

# Maximum linesearch stepsize
# TUNING! Too large or too small will slow convergence or end in local minimum...
tmax1_all = Nterms_all .+1    
# tmax1_all = Nterms_all    
# tmax1_all = 2.5*ones(length(Nterms_all))

# Run CG for each number of windows
objhist = []
gkkhist = []
resids = []
for i=1:length(Nterms_all)
    obj, gkk, res = runCG(Nterms_all[i], tmax1_all[i])
    push!(objhist, obj)
    push!(gkkhist, gkk)
    push!(resids, res)
end

# Plot convergence for each number of windows
i=1
pl_objhist = plot(objhist[i], yaxis=:log10, ylims=(1e-10,1e1), xlabel="Iteration", ylabel="Objective", label="Nterms="*string(Nterms_all[i]))
for i=2:length(objhist)
    global pl_objhist = plot!(objhist[i], yaxis=:log10, ylims=(1e-10,1e1), xlabel="Iteration", ylabel="Objective", label="Nterms="*string(Nterms_all[i]))
end
println("\n All convergence histories plotted in 'pl_objhist'.")

# Plot convergence for each number of windows (Gkk)
i=1
pl_gkkhist = plot(gkkhist[i], yaxis=:log10, ylims=(1e-10,1e1),  xlabel="Iteration", ylabel="Gkk", label="Nterms="*string(Nterms_all[i]))
for i=2:length(gkkhist)
    global pl_gkkhist = plot!(gkkhist[i], yaxis=:log10, ylims=(1e-10,1e1),  xlabel="Iteration", ylabel="Gkk", label="Nterms="*string(Nterms_all[i]))
end
println("\n All gradient histories plotted in 'pl_gkkhist'.")


# For the largest number of Nterms, plot the residuals per window, for the first few iterations iteration
plotniters = min(Niter, 50)
pl_residual = plot(resids[end][:,1:plotniters], yscale=:log10, legend=false, xlabel="window", ylabel="residual")
println(" Residuals per window for Nterms=", Nterms_all[end], " plotted in 'pl_residual'.")