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

    # final target
    Vtg = copy(Vvec[1])
    for q=2:nlen
        Vtg = Vvec[q]*Vtg
    end
    return Vtg
end

function init_intermediate_mat!(d::Int64, Wvec::Vector{Matrix{ComplexF64}}, Vvec::Vector{Matrix{ComplexF64}}, frobenius::Bool, amp::Float64=0.1)
    nlen = length(Wvec)
   
    S = Matrix(I, d,d)

    # if frobenius
    #     for q=1:nlen
    #         S = Vvec[q]*S
    #         Wvec[q] = S + rand(ComplexF64,d,d) * amp
    #     end
    # else
        for q=1:nlen
            # Exact propagator
            S = Vvec[q]*S

            # Perturbed initial condition
            H = im * log(S)
            Hperturb = rand(ComplexF64,d,d) * amp
            H += 0.5*(Hperturb + Hperturb')
            Wvec[q] = exp(-im*H)
        end
    # end
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


function eval_trace_obj(Zvec::Vector{Matrix{ComplexF64}}, Vvec::Vector{Matrix{ComplexF64}}, Vtg::Matrix{ComplexF64})
    # evaluate a sum of trace infidelities
    nlen = length(Zvec)
    d = size(Zvec[1],1)

    # Intermediate window residuals 
    G = sum(eval_trace_per_window(Zvec, Vvec)) / nlen
    # Final time target residual
    G += norm(Vtg)^2 - abs(tr(Vtg'*Vvec[nlen+1]*Zvec[nlen]))^2/d
    return G
end

function eval_frob_per_window(Zvec::Vector{Matrix{ComplexF64}}, Vvec::Vector{Matrix{ComplexF64}})
    nlen = length(Zvec)
    d = size(Zvec[1],1)

    G = zeros(nlen)
    G[1] = norm(Zvec[1]-Vvec[1])^2
    for q=2:nlen
        G[q] = norm(Zvec[q] - Vvec[q]*Zvec[q-1])^2
    end
    return G
end


function eval_frob_obj(Zvec::Vector{Matrix{ComplexF64}}, Vvec::Vector{Matrix{ComplexF64}}, Vtg::Matrix{ComplexF64})
    # evaluate a sum of trace infidelities
    nlen = length(Zvec)
    d = size(Zvec[1],1)

    # Intermediate window residuals 
    G = sum(eval_frob_per_window(Zvec, Vvec)) / nlen
    # Final time target residual
    G += norm(Zvec[nlen])^2 - abs(tr(Vtg'*Vvec[nlen+1]*Zvec[nlen]))^2/d
    return G
end


function frob_obj_line(t::Float64, Wvec::Vector{Matrix{ComplexF64}}, Hvec::Vector{Matrix{ComplexF64}}, Vvec::Vector{Matrix{ComplexF64}}, Uvec::Vector{Matrix{ComplexF64}}, Vtg::Matrix{ComplexF64})
    nTerms = length(Wvec) 
    for q=1:nTerms
        # Uvec[q] = exp(t * Hvec[q]) * Wvec[q] 
        Uvec[q] = Wvec[q] + t*Hvec[q]
    end
    return eval_frob_obj(Uvec, Vvec, Vtg)
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

function euclidean_extrapolation!(t::Float64, Wvec::Vector{Matrix{ComplexF64}}, Hvec::Vector{Matrix{ComplexF64}}, Uvec::Vector{Matrix{ComplexF64}})
    # evaluate the retraction at parameter 't'
    nTerms = length(Wvec)
    for q=1:nTerms
        Uvec[q] = Wvec[q] + t * Hvec[q]
    end
end


function eval_Euclidean_frob_grad!(Zvec::Vector{Matrix{ComplexF64}}, Vvec::Vector{Matrix{ComplexF64}}, Vtg::Matrix{ComplexF64}, Gvec::Vector{Matrix{ComplexF64}})
    # in-place results in Gvec
    Nterms = length(Zvec)
    d = size(Zvec[1],1)

    Gvec[1] = 2*(Zvec[1] - Vvec[1]) / Nterms

    for q=2:Nterms
        Gvec[q-1] += 2*(Zvec[q-1] - Vvec[q]'*Zvec[q] ) / Nterms # contributions to the previous gradient from Zvec[q]
        Gvec[q] = 2*(Zvec[q] - Vvec[q]*Zvec[q-1]) / Nterms
    end

    Gvec[Nterms] += 2*Zvec[Nterms] - 2*(Vvec[Nterms+1]'*Vtg*tr(Vtg'*Vvec[Nterms+1]*Zvec[Nterms])/d)

    Gvec[:] = Gvec
end



function eval_Euclidean_grad!(Zvec::Vector{Matrix{ComplexF64}}, Vvec::Vector{Matrix{ComplexF64}}, Vtg::Matrix{ComplexF64},  Gvec::Vector{Matrix{ComplexF64}})
    # in-place results in Gvec
    Nterms = length(Zvec)
    d = size(Zvec[1],1)

    Gvec[1] = 2*(Zvec[1] - Vvec[1]*tr(Vvec[1]'*Zvec[1])/d) / Nterms

    for q=2:Nterms
        Gvec[q-1] -= 2*(Vvec[q]'*Zvec[q] *tr(Zvec[q]'*Vvec[q]*Zvec[q-1])/d) / Nterms # contributions to the previous gradient from Zvec[q]
        Gvec[q] = 2*(Zvec[q] - Vvec[q]*Zvec[q-1]*tr(Zvec[q-1]'*Vvec[q]'*Zvec[q])/d) / Nterms
    end

    # Contribution from target in the last window
    q = Nterms+1
    Gvec[q-1] -= 2*(Vvec[q]'*Vtg*tr(Vtg'*Vvec[q]*Zvec[q-1])/d)

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

# Optimize with CG for given number of windows (Nterms) and maximum linesearch 'tmax'
function runCG(Nqubit::Int64, Nterms::Int64, tmax1::Float64, frobenius::Bool=true, amp::Float64=0.1)

    d = 2^Nqubit # dimension of matrices (d x d)
    Id = Matrix(I,d,d)

    # set the seed in the random number generator
    Random.seed!(1234)

    Vvec = Vector{Matrix{ComplexF64}}(undef,Nterms+1) # Propagators in each window: Nterms + 1 elements
    Wvec = Vector{Matrix{ComplexF64}}(undef,Nterms)   # unknown intermediate initial conditions,
    Gvec = Vector{Matrix{ComplexF64}}(undef,Nterms)   # Euclidean gradient
    Svec = Vector{Matrix{ComplexF64}}(undef,Nterms)   # Riemannian gradient (current point)
    Nvec = Vector{Matrix{ComplexF64}}(undef,Nterms)   # Riemannian gradient (next point)
    Hvec = Vector{Matrix{ComplexF64}}(undef,Nterms)   # Search direction
    Uvec = Vector{Matrix{ComplexF64}}(undef,Nterms)   # workspace

    # Set solution operators for each window and final target
    Vtg = init_random_unitary_mat!(d, Vvec)
    println("Final target dim: ", size(Vtg,1), " norm^2: ", norm(Vtg)^2)

    # Initial guess for intermediate states W: Small random perturbation from exact propagator
    init_intermediate_mat!(d, Wvec, Vvec, frobenius, amp)

    G0 = eval_trace_obj(Wvec, Vvec, Vtg)
    println("Initial objective: ", G0)

    # Euclidean gradient
    if frobenius
        eval_Euclidean_frob_grad!(Wvec, Vvec, Vtg, Gvec)

        Svec[:] = Gvec[:]
    else
        eval_Euclidean_grad!(Wvec, Vvec, Vtg, Gvec)

        # skew-Hermitian matrices from the Euclidian gradient
        for q=1:Nterms
            Svec[q] = skew(Gvec[q]*Wvec[q]')
        end
    end

    # ## Finite difference test
    # EPS = 1e-5
    # for q=1:Nterms
    #     Hvec[q] = rand(ComplexF64, d,d)
    # end
    # obj0 = frob_obj_line(-EPS, Wvec, Hvec, Vvec, Uvec, Vtg)
    # obj1 = frob_obj_line(EPS, Wvec, Hvec, Vvec, Uvec, Vtg)
    # fd_obj = (obj1-obj0) /(2*EPS)
    # dir_obj = inner_prod(Gvec, Hvec)
    # println("FD = ", fd_obj, " grad = ", dir_obj)
    # println("rel. error = ", abs(dir_obj - fd_obj)/abs(dir_obj))
    # stop

    # The initial search direction equals S
    global γ = 0.0
    Hvec[:] = Svec[:]

    # Initial objective function value
    # Define line search objective function with fixed direction
    if frobenius
        to_init = frob_obj_line(0.0, Wvec, Hvec, Vvec, Uvec, Vtg)
        ofunc3_f(t) =  frob_obj_line(-t, Wvec, Hvec, Vvec, Uvec, Vtg) 
    else
        to_init = trace_obj_line(0.0, Wvec, Hvec, Vvec, Uvec, Vtg)
        ofunc3(t) =  trace_obj_line(-t, Wvec, Hvec, Vvec, Uvec, Vtg) 
    end
    println("initial guess, objective (G) = ", to_init, " t=0.0")

    max_iter = 500
    global obj_hist = zeros(max_iter)
    global Gkk_hist = zeros(max_iter+1)
    global residuals = zeros(Nterms, max_iter)
    global Niter = 0

    # CG iterations with line-search
    for iter = 0:max_iter-1

        Niter = iter
        Gkk = inner_prod(Svec, Svec)
        Gkk_hist[iter+1] = Gkk

        if Gkk < 1e-10
            println("Found local minima after ", Niter, " CG iterations, Gkk = ", Gkk)
            break
        end

        # line search
        if frobenius
            result = optimize(ofunc3_f, -tmax1, tmax1, GoldenSection())
        else
            result = optimize(ofunc3, -tmax1, tmax1, GoldenSection())
        end

        t_min = Optim.minimizer(result)
        o_min = Optim.minimum(result)
        obj_hist[iter+1] = abs(o_min)

        println("iter = ", iter, " Gkk = ", Gkk, " γ = ", γ, ", minimum (G) = ", o_min, " minimizer (t): ", t_min)

        # next W:
        if frobenius
            euclidean_extrapolation!(-t_min, Wvec, Hvec, Uvec)
        else
            unitary_retraction!(-t_min, Wvec, Hvec, Uvec)
        end
        Wvec[:] = Uvec[:]

        # Store residuals per window and eval new gradient
        if frobenius
            residuals[:, iter+1]  = eval_frob_per_window(Wvec, Vvec)
            eval_Euclidean_frob_grad!(Wvec, Vvec, Vtg, Gvec)
            Nvec[:] = Gvec[:]
        else
            residuals[:, iter+1]  = eval_trace_per_window(Wvec, Vvec)
            eval_Euclidean_grad!(Wvec, Vvec, Vtg, Gvec)

            # form the skew-Hermitian matrices from the Euclidian gradients
            for q=1:Nterms
                Nvec[q] = skew(Gvec[q]*Wvec[q]')
            end
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

    unitary_err = 0.0
    Id = Matrix(I,d,d)
    for q=1:Nterms
        unitary_err += norm(Id - Wvec[q]'*Wvec[q])
    end
    unitary_err /= Nterms

    return obj_hist[1:Niter], Gkk_hist[1:Niter+1], residuals[:,1:Niter], unitary_err # Gkk_hist has Niter+1 elements
end # of function runCG


##########
# Run CG on increasing numbers of windows
##########

# varying the number of qubits
Nqubit_all = [2, 3, 4] # Dimension of unitary, dim = 2^q, where q is the number of qubits 
Nterms_all = [32, 32, 32] # Number of windows in the objective

# varying the number of windows
# Nqubit_all = [2, 2, 2, 2, 2] # Dimension of unitary, dim = 2^q, where q is the number of qubits 
# Nterms_all = [2, 4, 8, 16, 32] # Number of windows (-1) in the objective

# Nterms_all = [2,3,4,5,6]

# Maximum linesearch stepsize
# TUNING! Too large or too small will slow convergence or end in local minimum...
tmax1_all = Nterms_all .+ 1.0    
# tmax1_all = Nterms_all    
# tmax1_all = 2.5

# Switch between trace riemann CG (false) and frobenius (true)
frob = false # true # false # true

# Amplitude for perturbing the initial guess from the exact solution
init_amp = 1e-2 # 1e-1 # 

# Run CG for each number of windows
objhist = []
gkkhist = []
resids = []
unitary_err = []
for i=1:length(Nterms_all)
    obj, gkk, res, uerr = runCG(Nqubit_all[i], Nterms_all[i], tmax1_all[i], frob, init_amp)
    push!(objhist, obj)
    push!(gkkhist, gkk)
    push!(resids, res)
    push!(unitary_err, uerr)
end

# title string for the plots
if frob
    tstr = @sprintf("Frobenius norm & Euclidean grads, init_amp: %5.2f", init_amp)
else
    tstr = @sprintf("Trace infidelity & Riemannian grads, init_amp: %5.2f", init_amp)
end

# Plot convergence for each number of windows
pl_objhist = plot(yaxis=:log10, ylims=(1e-11,1e1), xlabel="Iteration", ylabel="Objective", title=tstr)
for i=1:length(objhist)
    lab_str = "Nq="*string(Nqubit_all[i])*" Nwin="*string(Nterms_all[i]+1)
    global pl_objhist = plot!(objhist[i], label=lab_str)
end
println("\n All convergence histories plotted in 'pl_objhist'.")

# Plot convergence for each number of windows (Gkk)
pl_gkkhist = plot(yaxis=:log10, ylims=(1e-11,1e1),  xlabel="Iteration", ylabel="Norm^2(grad)", title=tstr)
for i=1:length(gkkhist)
    lab_str = "Nq="*string(Nqubit_all[i])*" Nwin="*string(Nterms_all[i]+1)
    global pl_gkkhist = plot!(gkkhist[i], label=lab_str)
end
println("\n All gradient histories plotted in 'pl_gkkhist'.")


# For the largest number of Nterms, plot the residuals per window, for the first few iterations iteration
if frob
    tstr = @sprintf("Window mismatch, Frobenius, Nq=%d, Nwin=%d, init_amp: %5.2f", Nqubit_all[end], Nterms_all[end], init_amp)
else
    tstr = @sprintf("Window mismatch, Infidelity, Nq=%d, Nwin=%d, init_amp: %5.2f", Nqubit_all[end], Nterms_all[end], init_amp)
end
plotniters = size(resids[end],2)
pl_residual = plot(yscale=:log10, ylims=(1e-11,1e1), xlabel="window", ylabel="residual", title=tstr, leg=:outerright, size=(800,400))
for i=1:5:plotniters
    lab_str = "iter="*string(i)
    plot!(pl_residual, resids[end][:,i], lab=lab_str)
end
println("\n Residuals per window for Nwin=", Nterms_all[end]+1, " plotted in 'pl_residual'.")