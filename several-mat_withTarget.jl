using LinearAlgebra
using Plots
using Printf
using Random
using Optim

skew(Z) = 0.5*(Z - Z') # skew-Hermitian part of Z

function generate_solution!(d::Int64, Vvec::Vector{Matrix{ComplexF64}}, Vvec_Ham::Vector{Matrix{ComplexF64}}, alpha::Vector{Float64})
    nlen = length(Vvec)

    # Random optimal control 
    alpha = rand(nlen+1)

    # Random propergator Hamiltonians per window
    for q=1:nlen
        Vvec_Ham[q] = rand(ComplexF64,d,d)
        Vvec_Ham[q] = 0.5*(Vvec_Ham[q] + Vvec_Ham[q]')
    end

    # Evaluate solution operators per window
    eval_propagators!(alpha, Vvec_Ham, Vvec)

    # Final target
    Vtg = copy(Vvec[1])
    for q=2:nlen
        Vtg = Vvec[q]*Vtg
    end
    return Vtg
end

function eval_propagators!(alpha::Vector{Float64}, Vvec_Ham::Vector{Matrix{ComplexF64}}, Vvec::Vector{Matrix{ComplexF64}})
    for  q=1:length(Vvec)
        Vvec[q] = exp(-im*alpha[q]*Vvec_Ham[q])
    end
end

function init_intermediate_mat!(d::Int64, Wvec::Vector{Matrix{ComplexF64}}, Vvec::Vector{Matrix{ComplexF64}}, frobenius)
    nlen = length(Wvec)
   
    S = Matrix(I, d,d)
    amp = 1e-1

    if frobenius
        for q=1:nlen
            S = Vvec[q]*S
            Wvec[q] = S + rand(ComplexF64,d,d) * amp
        end
    else
        for q=1:nlen
            # Exact propagator
            S = Vvec[q]*S

            # Perturbed initial condition
            H = im * log(S)
            Hperturb = rand(ComplexF64,d,d) * amp
            H += 0.5*(Hperturb + Hperturb')
            Wvec[q] = exp(-im*H)
        end
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


function eval_trace_obj!(Zvec::Vector{Matrix{ComplexF64}}, alpha::Vector{Float64}, Vvec_Ham::Vector{Matrix{ComplexF64}}, Vtg::Matrix{ComplexF64}, Vvec::Vector{Matrix{ComplexF64}})
    # evaluate a sum of trace infidelities
    nlen = length(Zvec)
    d = size(Zvec[1],1)

    # Evaluate current propagators 
    eval_propagators!(alpha, Vvec_Ham, Vvec)
    # Sum up intermediate window residuals 
    G = sum(eval_trace_per_window(Zvec, Vvec)) / nlen
    # Add final time target mismatch
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


function eval_frob_obj!(Zvec::Vector{Matrix{ComplexF64}}, alpha::Vector{Float64}, Vvec_Ham::Vector{Matrix{ComplexF64}}, Vtg::Matrix{ComplexF64}, Vvec::Vector{Matrix{ComplexF64}}, )
    # evaluate a sum of trace infidelities
    nlen = length(Zvec)
    d = size(Zvec[1],1)

    # Evaluate current propagators 
    eval_propagators!(alpha, Vvec_Ham, Vvec)
    # Sum up intermediate window residuals 
    G = sum(eval_frob_per_window(Zvec, Vvec)) / nlen
    # Add final time target mismatch
    G += norm(Zvec[nlen])^2 - abs(tr(Vtg'*Vvec[nlen+1]*Zvec[nlen]))^2/d
    return G
end

function frob_obj_line(t::Float64, Wvec::Vector{Matrix{ComplexF64}}, alpha::Vector{Float64}, Vvec_Ham::Vector{Matrix{ComplexF64}}, Hvec::Vector{Matrix{ComplexF64}}, Halpha::Vector{Float64}, Vvec::Vector{Matrix{ComplexF64}}, Uvec::Vector{Matrix{ComplexF64}}, Ualpha::Vector{Float64}, Vtg::Matrix{ComplexF64})
    nTerms = length(Wvec) 
    for q=1:nTerms
        # Uvec[q] = exp(t * Hvec[q]) * Wvec[q] 
        Uvec[q] = Wvec[q] + t*Hvec[q]
    end
    Ualpha[:] = alpha[:] .+ t*Halpha[:]
    return eval_frob_obj!(Uvec, Ualpha, Vvec_Ham, Vtg, Vvec)
end

function trace_obj_line(t::Float64, Wvec::Vector{Matrix{ComplexF64}}, alpha::Vector{Float64}, Vvec_Ham::Vector{Matrix{ComplexF64}}, Hvec::Vector{Matrix{ComplexF64}}, Halpha::Vector{Float64}, Vvec::Vector{Matrix{ComplexF64}}, Uvec::Vector{Matrix{ComplexF64}}, Ualpha::Vector{Float64}, Vtg::Matrix{ComplexF64})
    nTerms = length(Wvec) 
    for q=1:nTerms
        Uvec[q] = exp(t * Hvec[q]) * Wvec[q] 
        # Uvec[q] = Wvec[q] + t*Hvec[q] 
    end
    Ualpha[:] = alpha[:] .+ t*Halpha[:]
    return eval_trace_obj!(Uvec, Ualpha, Vvec_Ham, Vtg, Vvec)
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


function eval_Euclidean_frob_grad!(Zvec::Vector{Matrix{ComplexF64}}, alpha::Vector{Float64}, Vvec::Vector{Matrix{ComplexF64}}, Vvec_Ham::Vector{Matrix{ComplexF64}}, Vtg::Matrix{ComplexF64}, Gvec::Vector{Matrix{ComplexF64}}, Galpha::Vector{Float64})
    # Gradient in-place in Gvec and Galpha
    Nterms = length(Zvec)
    d = size(Zvec[1],1)

    # Just to make sure Vvec is set for current alpha, re-evaluate them here.
    eval_propagators!(alpha, Vvec_Ham, Vvec)

    # Gradient wrt intermediate states
    Gvec[1] = 2*(Zvec[1] - Vvec[1]) / Nterms
    for q=2:Nterms
        Gvec[q-1] += 2*(Zvec[q-1] - Vvec[q]'*Zvec[q] ) / Nterms # contributions to the previous gradient from Zvec[q]
        Gvec[q] = 2*(Zvec[q] - Vvec[q]*Zvec[q-1]) / Nterms
    end
    Gvec[Nterms] += 2*Zvec[Nterms] - 2*(Vvec[Nterms+1]'*Vtg*tr(Vtg'*Vvec[Nterms+1]*Zvec[Nterms])/d)
    Gvec[:] = Gvec

    # Gradient with respect to alpha
    Vprime = -im*Vvec_Ham[1]*Vvec[1]
    Galpha[1] = -2*real(tr((Zvec[1] - Vvec[1])'*Vprime)) / Nterms
    for q=2:Nterms
        Vprime = -im*Vvec_Ham[q]*Vvec[q]
        Galpha[q] = -2*real(tr((Zvec[q] - Vvec[q]*Zvec[q-1])'*Vprime*Zvec[q-1])) / Nterms
    end
    q = Nterms+1
    Vprime = -im*Vvec_Ham[q]*Vvec[q]
    Galpha[q] = -2*real(tr(Zvec[q-1]'*Vvec[q]'*Vtg) * tr(Vtg'*Vprime*Zvec[q-1]))/d
end


function eval_Euclidean_trace_grad!(Zvec::Vector{Matrix{ComplexF64}}, alpha::Vector{Float64}, Vvec::Vector{Matrix{ComplexF64}}, Vvec_Ham::Vector{Matrix{ComplexF64}}, Vtg::Matrix{ComplexF64}, Gvec::Vector{Matrix{ComplexF64}}, Galpha::Vector{Float64})

    # in-place results in Gvec
    Nterms = length(Zvec)
    d = size(Zvec[1],1)

    # Just to make sure Vvec is set for current alpha, re-evaluate them here.
    eval_propagators!(alpha, Vvec_Ham, Vvec)
    
    # Gradient wrt intermediate states
    Gvec[1] = 2*(Zvec[1] - Vvec[1]*tr(Vvec[1]'*Zvec[1])/d) / Nterms
    for q=2:Nterms
        Gvec[q-1] -= 2*(Vvec[q]'*Zvec[q] *tr(Zvec[q]'*Vvec[q]*Zvec[q-1])/d) / Nterms # contributions to the previous gradient from Zvec[q]
        Gvec[q] = 2*(Zvec[q] - Vvec[q]*Zvec[q-1]*tr(Zvec[q-1]'*Vvec[q]'*Zvec[q])/d) / Nterms
    end
    # Contribution from target in the last window
    q = Nterms+1
    Gvec[q-1] -= 2*(Vvec[q]'*Vtg*tr(Vtg'*Vvec[q]*Zvec[q-1])/d)
    Gvec[:] = Gvec

    # Gradient with respect to alpha
    Vprime = -im*Vvec_Ham[1]*Vvec[1]
    Galpha[1] = -2*real(tr(Vvec[1]'*Zvec[1]) * tr(Zvec[1]'*Vprime))/d/Nterms
    for q=2:Nterms
        Vprime = -im*Vvec_Ham[q]*Vvec[q]
        Galpha[q] = -2*real(tr(Zvec[q-1]'*Vvec[q]'*Zvec[q]) * tr(Zvec[q]'*Vprime*Zvec[q-1]))/d/Nterms
    end
    q = Nterms+1
    Vprime = -im*Vvec_Ham[q]*Vvec[q]
    Galpha[q] = -2*real(tr(Zvec[q-1]'*Vvec[q]'*Vtg) * tr(Vtg'*Vprime*Zvec[q-1]))/d
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
function runCG(Nterms, tmax1, frobenius=true)
# Nterms = 4
# tmax1 = 2
# frobenius = true

    # set the seed in the random number generator
    Random.seed!(1234)

    Vvec = Vector{Matrix{ComplexF64}}(undef,Nterms+1) # Propagators in each window: Nterms + 1 elements
    Vvec_Ham = Vector{Matrix{ComplexF64}}(undef,Nterms+1) # Hamiltonian for each Propagator
    Wvec = Vector{Matrix{ComplexF64}}(undef,Nterms)   # unknown intermediate initial conditions,
    Gvec = Vector{Matrix{ComplexF64}}(undef,Nterms)   # Euclidean gradient
    Svec = Vector{Matrix{ComplexF64}}(undef,Nterms)   # Riemannian gradient (current point)
    Nvec = Vector{Matrix{ComplexF64}}(undef,Nterms)   # Riemannian gradient (next point)
    Hvec = Vector{Matrix{ComplexF64}}(undef,Nterms)   # Search direction
    Uvec = Vector{Matrix{ComplexF64}}(undef,Nterms)   # workspace
    alpha = zeros(Float64,Nterms+1)                 # control parameters
    Galpha = zeros(Float64,Nterms+1)                # Gradient of controls
    Salpha = zeros(Float64,Nterms+1)                # 
    Nalpha = zeros(Float64,Nterms+1)                # 
    Halpha = zeros(Float64,Nterms+1)                # 
    Ualpha = zeros(Float64,Nterms+1)                # 

    # General optimal controls, solution operators and final target
    Vtg = generate_solution!(d, Vvec, Vvec_Ham, alpha)
    alpha_opt = copy(alpha)
    println("Final target dim: ", size(Vtg,1), " norm^2: ", norm(Vtg)^2)

    # Initial guess for intermediate states W: Small random perturbation from exact propagator
    init_intermediate_mat!(d, Wvec, Vvec, frobenius)

    # Initial guess for the control parameters
    amp = 0.1
    alpha[:] = alpha_opt + amp*rand(Nterms+1)
    # alpha[:] = rand(Nterms+1)

    if frobenius
        G0 = eval_frob_obj!(Wvec, alpha, Vvec_Ham, Vtg, Vvec)
    else
        G0 = eval_trace_obj!(Wvec, alpha, Vvec_Ham, Vtg, Vvec)
    end
    println("Initial objective: ", G0)

    # Euclidean gradient
    if frobenius
        eval_Euclidean_frob_grad!(Wvec, alpha, Vvec, Vvec_Ham, Vtg, Gvec, Galpha)
        Svec[:] = Gvec[:]
        Salpha[:] = Galpha[:]
    else
        eval_Euclidean_trace_grad!(Wvec, alpha, Vvec, Vvec_Ham, Vtg, Gvec, Galpha)
        # skew-Hermitian matrices from the Euclidian gradient
        for q=1:Nterms
            Svec[q] = skew(Gvec[q]*Wvec[q]')
        end
        Salpha[:] = Galpha[:]
    end

    # ## Finite difference test
    # Random directional derivatives
    EPS = 1e-5
    for i=345:355
        Random.seed!(i)
        for q=1:Nterms
            # Hvec[q] =0.0* rand(ComplexF64, d,d)
            Hvec[q] = rand(ComplexF64, d,d)
        end
        # Halpha = 0.0*rand(Float64, length(alpha))
        Halpha = rand(Float64, length(alpha))
        if frobenius
            obj0 = frob_obj_line(-EPS, Wvec, alpha, Vvec_Ham, Hvec, Halpha, Vvec, Uvec, Ualpha, Vtg)
            obj1 = frob_obj_line(EPS, Wvec, alpha, Vvec_Ham, Hvec, Halpha, Vvec, Uvec, Ualpha, Vtg)
            GvecR = copy(Gvec)
        else
            for q=1:Nterms # Need swew symm part for Riemann gradient
                Hvec[q] = skew(Hvec[q]*Wvec[q]')
            end
            obj0 = trace_obj_line(-EPS, Wvec, alpha, Vvec_Ham, Hvec, Halpha, Vvec, Uvec, Ualpha, Vtg)
            obj1 = trace_obj_line(EPS, Wvec, alpha, Vvec_Ham, Hvec, Halpha, Vvec, Uvec, Ualpha, Vtg)
            # Riemann gradient ?
            GvecR = copy(Gvec)
            for q=1:Nterms
                GvecR[q]=skew(GvecR[q]*Wvec[q]') # Missing a Wvec after skew()?
            end
        end
        fd_obj = (obj1-obj0) /(2*EPS)
        # dir_obj = inner_prod(Gvec, Hvec) + Galpha'*Halpha
        dir_obj = inner_prod(GvecR, Hvec) + Galpha'*Halpha
        println("obj=", obj0," ", obj1, ", FD = ", fd_obj, " grad = ", dir_obj, " FD rel. error = ", abs(dir_obj - fd_obj)/abs(dir_obj))
    end
    stop
    # # # Separately: Grad with respect to alpha
    # alphaP = copy(alpha)
    # # for i=1:length(alpha)
    # #     alphaP[i] = alpha[i] - EPS
    # #     if frobenius
    # #         obj0 = eval_frob_obj!(Wvec, alphaP, Vvec_Ham, Vtg, Vvec)
    # #     else
    # #         obj0 = eval_trace_obj!(Wvec, alphaP, Vvec_Ham, Vtg, Vvec)
    # #     end
    # #     alphaP[i] = alpha[i] + EPS
    # #     if frobenius
    # #         obj1 = eval_frob_obj!(Wvec, alphaP, Vvec_Ham, Vtg, Vvec)
    # #     else
    # #         obj1 = eval_trace_obj!(Wvec, alphaP, Vvec_Ham, Vtg, Vvec)
    # #     end
    # #     alphaP[i] = alpha[i] # Reset for next iteration

    # #     fd_obj = (obj1-obj0) /(2*EPS)
    # #     dir_obj = Galpha[i]
    # #     # println("FD = ", fd_obj, " grad = ", dir_obj)
    # #     println("FD rel. error = ", abs(dir_obj - fd_obj)/abs(dir_obj))
    # # end
    # # stop

    # The initial search direction equals S
    global γ = 0.0
    global gg = 0.0
    Hvec[:] = Svec[:]
    Halpha[:] = Salpha[:]

    # Initial objective function value
    # Define line search objective function with fixed direction
    if frobenius
        to_init = frob_obj_line(0.0, Wvec, alpha, Vvec_Ham, Hvec, Halpha, Vvec, Uvec, Ualpha, Vtg)
        ofunc3_f(t) = frob_obj_line(-t, Wvec, alpha, Vvec_Ham, Hvec, Halpha, Vvec, Uvec, Ualpha, Vtg) 
    else
        to_init = trace_obj_line(0.0, Wvec, alpha, Vvec_Ham, Hvec, Halpha, Vvec, Uvec, Ualpha, Vtg)
        ofunc3(t) =  trace_obj_line(-t, Wvec, alpha, Vvec_Ham, Hvec, Halpha, Vvec, Uvec, Ualpha, Vtg) 
    end
    println("initial guess, objective (G) = ", to_init, " t=0.0")


    plo = plot(leg=:outerright, size=(800,400), xaxis = "t", yaxis = "objective", ylims=(0,3.0))

    max_iter = 400
    global obj_hist = zeros(max_iter)
    global Gkk_hist = zeros(max_iter+1)
    global residuals = zeros(Nterms, max_iter)
    global Niter = 0

    # CG iterations with line-search
    for iter = 0:max_iter-1

        Niter = iter
        Gkk = inner_prod(Svec, Svec)
        Gkk_alpha = Salpha'*Salpha
        Gkk_hist[iter+1] = Gkk

        if Gkk+Gkk_alpha < 1e-9
            println("Found local minima after ", Niter, " CG iterations, Gkk = ", Gkk, "Gkk_alpha=", Gkk_alpha)
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

        println("iter = ", iter, " Gkk = ", Gkk+Gkk_alpha, " γ = ", γ+gg, ", minimum (G) = ", o_min, " minimizer (t): ", t_min)

        tls = LinRange(0.0, tmax1, 101)
        # op2 = ofunc3_f.(tls) # Follow the unitary retraction (for plotting)
        op2 = ofunc3.(tls) # Follow the unitary retraction (for plotting)
        if iter <20
            lstr = "frob-"*string(iter)
            plot!(plo, tls, op2, lab=lstr, lw=3)
            scatter!(plo, [t_min], [o_min], lab=:false)
        end


        # next W:
        if frobenius
            euclidean_extrapolation!(-t_min, Wvec, Hvec, Uvec)
        else
            unitary_retraction!(-t_min, Wvec, Hvec, Uvec)
        end
        Wvec[:] = Uvec[:]

        # Next alpha
        alpha[:] -= t_min*Halpha[:]
        eval_propagators!(alpha, Vvec_Ham, Vvec)

        # Store residuals per window and eval new gradient
        if frobenius
            residuals[:, iter+1]  = eval_frob_per_window(Wvec, Vvec)
            eval_Euclidean_frob_grad!(Wvec, alpha, Vvec, Vvec_Ham, Vtg, Gvec, Galpha)
            Nvec[:] = Gvec[:]
        else
            residuals[:, iter+1]  = eval_trace_per_window(Wvec, Vvec)
            eval_Euclidean_trace_grad!(Wvec, alpha, Vvec, Vvec_Ham, Vtg, Gvec, Galpha)

            # form the skew-Hermitian matrices from the Euclidian gradients
            for q=1:Nterms
                Nvec[q] = skew(Gvec[q]*Wvec[q]')
            end
        end
        Nalpha[:] = Galpha[:]

        # next search direction
        Uvec[:] = Nvec[:] - Svec[:]
        Ualpha[:] = Nalpha[:]-Salpha[:]
        γ = inner_prod(Uvec, Nvec)/Gkk
        g =  Ualpha'*Nalpha[:] / Gkk_alpha
        gg = (inner_prod(Uvec, Nvec) + Ualpha'*Nalpha[:]) / (Gkk + Gkk_alpha)
        Hvec[:] = Nvec[:] + gg .* Hvec[:]
        Halpha[:] = Nalpha[:] + gg .*Halpha[:]
        # Hvec[:] = Nvec[:] + γ .* Hvec[:]
        # Halpha[:] = Nalpha[:] + g *Halpha[:]

        HGprod =inner_prod(Hvec, Nvec) + Halpha'*Nalpha
        if HGprod < 0.0
        # if true
            println("HGprod = ", HGprod, " < 0, taking a steepest descent direction")
            Hvec[:] = Nvec[:]
            Halpha[:] = Nalpha[:]
        end
        # HGprodg =  Halpha'*Nalpha
        # if HGprodg < 0.0
        #     println("HGprodg = ", HGprodg, " < 0, taking a steepest descent direction")
        #     Halpha[:] = Nalpha[:]
        # end

        # save new descent direction
        Svec[:] = Nvec[:]
        Salpha[:] = Nalpha[:]
    end

    unitary_err = 0.0
    Id = Matrix(I,d,d)
    for q=1:Nterms
        unitary_err += norm(Id - Wvec[q]'*Wvec[q])
    end
    unitary_err /= Nterms

    return obj_hist[1:Niter], Gkk_hist[1:Niter+1], residuals[:,1:Niter], unitary_err, plo # Gkk_hist has Niter+1 elements
end # of function runCG


##########
# Run CG on increasing numbers of windows
##########

# Number of windows in the objective
# Nterms_all = [2,3,4,5,6]
# Nterms_all = [2, 4, 8, 16, 32]
Nterms_all = [2, 4, 8, 16]
# Nterms_all = [4]

# Maximum linesearch stepsize
# TUNING! Too large or too small will slow convergence or end in local minimum...
tmax1_all = Nterms_all .+1    
# tmax1_all = Nterms_all    
# tmax1_all = 2.5*ones(length(Nterms_all))

# Switch between trace riemann CG (false) and frobenius (true)
frob = false

# Run CG for each number of windows
objhist = []
gkkhist = []
resids = []
unitary_err = []
plos = []
for i=1:length(Nterms_all)
    obj, gkk, res, uerr, plo = runCG(Nterms_all[i], tmax1_all[i], frob)
    push!(objhist, obj)
    push!(gkkhist, gkk)
    push!(resids, res)
    push!(unitary_err, uerr)
    push!(plos, plo)
end
println("Line search plot object stored in variable 'plos'")

# title string for the plots
if frob
    tstr = "Frobenius norm & Euclidean grads"
else
    tstr = "Trace infidelity & Riemannian grads"
end

# Plot convergence for each number of windows
i=1
pl_objhist = plot(objhist[i], yaxis=:log10, ylims=(1e-10,1e1), xlabel="Iteration", ylabel="Objective", label="Nterms="*string(Nterms_all[i]), title=tstr)
for i=2:length(objhist)
    global pl_objhist = plot!(objhist[i], yaxis=:log10, ylims=(1e-10,1e1), xlabel="Iteration", ylabel="Objective", label="Nterms="*string(Nterms_all[i]))
end
println("\n All convergence histories plotted in 'pl_objhist'.")

# Plot convergence for each number of windows (Gkk)
i=1
pl_gkkhist = plot(gkkhist[i], yaxis=:log10, ylims=(1e-10,1e1),  xlabel="Iteration", ylabel="Gkk", label="Nterms="*string(Nterms_all[i]), title=tstr)
for i=2:length(gkkhist)
    global pl_gkkhist = plot!(gkkhist[i], yaxis=:log10, ylims=(1e-10,1e1),  xlabel="Iteration", ylabel="Gkk", label="Nterms="*string(Nterms_all[i]))
end
println("\n All gradient histories plotted in 'pl_gkkhist'.")


# For the largest number of Nterms, plot the residuals per window, for the first few iterations iteration
plotniters = min(Niter, 50)
pl_residual = plot(resids[end][:,1:5:plotniters], yscale=:log10, legend=true, xlabel="window", ylabel="residual", title=tstr)
println("\n Residuals per window for Nterms=", Nterms_all[end], " plotted in 'pl_residual'.")