using LinearAlgebra
using Plots
using Printf
using Random
using Optim

q = 2 # two qubits
d = 2^q # dimension of matrices (d x d)
Id = Matrix(I,d,d)
Random.seed!(1234)
skew(Z) = 0.5*(Z - Z') 

function runCG(nwins, tmax1)
    global d, Id, V, S, W, H, U,  γ, obj_hist
    global Niter

    # Construct solution operators for each window
    V = Vector{Matrix{ComplexF64}}(undef, nwins)
    for i=1:nwins
        Hi = rand(ComplexF64,d,d)
        Hsym = 0.5*(Hi + Hi')
        V[i] = exp(-im*Hsym)
    end

    # Initial guess for intermediate window states: identity matrices
    W = Vector{Matrix{ComplexF64}}(undef, nwins)
    for i=1:nwins
        W[i] = Matrix(I,d,d)
    end

    # scaling (only used in SD)
    t_fact = 1.0*ones(nwins)

    # Define the per-window objective functions and gradients
    trace_obj_vec = Vector{Function}(undef, nwins);
    gradient_vec = Vector{Function}(undef, nwins);
    for i=1:nwins
        # objective:
        if i==1
            func_win = (Z) ->  norm(Z[i])^2 - abs(tr(Z[i]'*V[i]))^2/d   # func maps the global vector Z[1], ..., Z[N] to the i-th window objective function
        else
            func_win = (Z) ->  norm(Z[i])^2 - abs(tr(Z[i]'*V[i]*Z[i-1]))^2/d 
        end
        trace_obj_vec[i] = func_win

        # gradient:
        if i==1
            function foo1(Z, i)
                global V, nwins
                g = [zeros(ComplexF64, d,d) for _ = 1:nwins] 
                g[i]=2*(Z[i] - V[i]*tr(V[i]'*Z[i])/d ) 
                return g 
            end
            grad_win = (Z) -> foo1(Z, i)
        else
            function foo2(Z, i)
                global V, nwins
                g = [zeros(ComplexF64, d,d) for _ = 1:nwins] 
                g[i-1] = 2*(-V[i]'*Z[i] *tr(Z[i]'*V[i]*Z[i-1])/d)
                g[i]   = 2*(Z[i] - V[i]*Z[i-1]*tr(Z[i-1]'*V[i]'*Z[i])/d)
                return g
            end
            grad_win = (Z) -> foo2(Z, i)
        end
        gradient_vec[i] = grad_win
    end

    # Define the overall objective function and graditne
    trace_obj(Z) = sum([h(Z) for h in trace_obj_vec]) / nwins
    gradient(Z) = sum([g(Z) for g in gradient_vec]) / nwins 

    # Initial guess for S and H
    S = Vector{Matrix{ComplexF64}}(undef, nwins)
    H = Vector{Matrix{ComplexF64}}(undef, nwins)
    for i=1:nwins
        S[i] = skew(gradient(W)[i]*W[i]')
        H[i] = S[i]
    end

    # Define unitary retractions at W in the directions H
    U = Vector{Function}(undef, nwins);
    for i=1:nwins
        U[i] = (t) -> exp(t_fact[i]*t * H[i]) * W[i]
    end

    # Define linesearch objective function
    ofunc_LS(t) =  trace_obj([u(-t) for u in U]) # objective function, fixed direction
    tstr = "Trace objective, unitary retraction"

    # Initial weight factor
    γ = 0.0

    # Initial objective function value
    to_init = ofunc_LS(0.0)
    println("initial guess, minimum (G) = ", to_init, " t=0.0")

    max_iter = 200
    obj_hist = zeros(max_iter)
    residuals = zeros(max_iter, nwins)
    Niter = 0
    # steepest descent with line-search, keeping ϕ fixed
    for iter = 0:max_iter-1
        global W, S, H, U,  γ, Niter, V, obj_hist

        Niter = iter
        Gkk = sum([real(tr(S[i]'*S[i])) for i in 1:nwins])

        if Gkk < 1e-9
            println("Found local minima after ", Niter, " CG iterations, Gkk = ", Gkk)
            break
        end

        # line search (algorithm doesn't make any difference)
        if iter < 2
            result = optimize(ofunc_LS, -tmax1, tmax1, Brent()) # start at t=0
        else
            result = optimize(ofunc_LS, -tmax1, tmax1, GoldenSection())
        end
        t_min = Optim.minimizer(result)
        o_min = Optim.minimum(result)
        obj_hist[iter+1] = o_min

        println("iter = ", iter, " Gkk = ", Gkk, " γ = ", γ, ", minimum (G) = ", o_min, " minimizer (t): ", t_min)

        # next W:
        for i=1:nwins
            W[i] = U[i](-t_min)
        end

        # Evaluate and store the residuals in each window
        residuals[iter+1, :]  = [trace_obj_vec[i](W) for i=1:nwins]

        # Steepest descent direction at new W
        N = Vector{Matrix{ComplexF64}}(undef, nwins)
        for i=1:nwins
            N[i] = skew(gradient(W)[i]*W[i]')
        end

        # next search direction
        γ = real(sum([tr((N[i]'-S[i]')*N[i]) for i=1:nwins])) / Gkk
        for i=1:nwins
            H[i] = N[i] + γ * H[i]
        end

        HGprod = real(sum([tr(H[i]'*N[i]) for i=1:nwins]))

        if HGprod < 0.0
            println("HGprod = ", HGprod, " < 0, using steepest descent instead ")
            for i=1:nwins
                H[i] = N[i]
            end
        end

        # save new descent direction
        for i=1:nwins
            S[i] = N[i]
        end
    end

    # Plot overall optimization history
    tstr = @sprintf("CG conv hist, %d matrices", nwins)
    plconv = plot(obj_hist[1:Niter], lab=:none, title=tstr, yaxis=:log10,ylims=(1e-10,1e1), xlabel="Iteration", ylabel="Objective")
    # println("Convergence plot in variable 'plconv'")

    # Plot window residuals per optimization iteration
    plwin = plot(residuals[1:Niter, :]', yscale=:log10, legend = false)
    # println("Residuals per window stored in variable 'plwin'")

    return obj_hist, plconv, plwin
end

# Set the Number of windows to start a CG iteration
nwins_all = [1,2,4,8,16,32,64]
# nwins_all = [1,2,3,4,5,6]

# Maximum linesearch step. 
# TUNING! Too small or too large slows down the convergence.
tmax1_all = [nwins_all[i] for i=1:length(nwins_all)]
# tmax1_all = [3.0, 3.0, 4.0, 4.0,5.0,6.0]

# Loop over number of windows, running the CG optimization 
objhist = []
plwin  = []
for i=1:length(nwins_all)

    global nwins = nwins_all[i]
    tmax1 = tmax1_all[i] 

    # Run CG
    println("\n Optimizing on ", nwins, " windows. Tmax=", tmax1)
    objhisti, plconvi, plwini = runCG(nwins, tmax1)

    # Store plots and history
    push!(objhist, objhisti)
    push!(plwin, plwini)

end
println("Per-window residuals stored in 'plwini' vector for each number of windows")
println("Optimization histories stored in 'objhist' for each number of windows")

# Set zero values to small number for plotting 
for i=1:length(objhist)
    objhist[i][abs.(objhist[i]) .== 0 ] .= 1e-14
end

# Plot convergence for each number of windows into one plot
i=1
pl_objhist = plot(objhist[i], yaxis=:log10, ylims=(1e-10,1e1), xlims=(1,Niter), xlabel="Iteration", ylabel="Objective", label="nwins="*string(nwins_all[i]))
for i=2:length(objhist)
    global pl_objhist = plot!(objhist[i], yaxis=:log10, ylims=(1e-10,1e1), xlims=(1,Niter), xlabel="Iteration", ylabel="Objective", label="nwins="*string(nwins_all[i]))
end
println("\n All convergence histories plotted in 'pl_objhist'.")