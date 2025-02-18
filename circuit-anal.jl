using LinearAlgebra
using Plots
using Printf
using Optim

f_t = 5.0 # GHz
f_r = 7.0 
ω_t = 2*pi*f_t # rad / ns
ω_r = 2*pi*f_r
Cg_o_Ct = 0.1 # dimensionless
Cg_o_Cr = 0.2

N_tr(s) = s*(ω_t^2 + s^2 * (1.0 + Cg_o_Ct)) # removed factor 1/Cr
D1_tr(s) = s^2*(ω_t^2 + s^2)
D2_tr(s) = (ω_r^2 + s^2)*(ω_t^2 + s^2)  
D3_tr(s) = s^2*(ω_r^2 + s^2)
D_tr(s) = Cg_o_Cr*D1_tr(s) + D2_tr(s) + Cg_o_Ct*D3_tr(s)
Z_tr(s) = (N_tr(s)/D_tr(s))

# real-valued version setting s^2 = -x^2
R1_tr(x) = -x^2*(ω_t^2 - x^2)
R2_tr(x) = (ω_r^2 - x^2)*(ω_t^2 - x^2)  
R3_tr(x) = -x^2*(ω_r^2 - x^2)
R_tr(x) = Cg_o_Cr*R1_tr(x) + R2_tr(x) + Cg_o_Ct*R3_tr(x)
obj(x) = R_tr(x)^2

x0 = ω_t
# find the zeros of R_tr
result = optimize(obj, x0-2*pi, x0)
f_t_corr = Optim.minimizer(result)*0.5/pi
println("ω_t (bare): ", f_t, " (coupled): ", f_t_corr, " GHz")

# 2nd root
x0 = ω_r
result = optimize(obj, x0-2*pi, x0)
f_r_corr = Optim.minimizer(result)*0.5/pi
println("ω_t (bare): ", f_r, " (coupled): ", f_r_corr, " GHz")


# define grid in 's'
Nω = 2000
ω_0 = 4.3 * 2*pi # (ω_t - 2*pi*2)
ω_1 = 7.1 * 2*pi # (ω_r + 2*pi*0.5)
ω_grid = LinRange(ω_0, ω_1, Nω)

tstr = @sprintf("Cg/Ct = %.1f, Cg/Cr = %.1f", Cg_o_Ct, Cg_o_Cr)
pl0 = plot(title = tstr, xaxis="freq [GHz]")
plot!(pl0, ω_grid*0.5/pi, real(N_tr.(im*ω_grid)), lab="Re{N(s)/s}")
scatter!(pl0, [f_t], [0.0], markershape=:cross, label="Trmn(bare)")
scatter!(pl0, [f_t_corr], [1e-2], markershape=:cross, label="trmn(cpld)")
scatter!(pl0, [f_r], [0.0], markershape=:cross, label="Res(bare)")
scatter!(pl0, [f_r_corr], [1e-2], markershape=:cross, label="res(cpld)")

pl1 = plot(title = tstr, xaxis="freq [GHz]")
plot!(pl1, ω_grid*0.5/pi, real(D_tr.(im*ω_grid)), lab="Re{D_tr(ω)}")
#scatter!(pl1, ω_grid*0.5/pi, R_tr.(ω_grid), lab="R_tr(ω)}")
#plot!(pl1, ω_grid*0.5/pi, real(D1_tr.(im*ω_grid)), lab="Re{D1(ω)}")
#plot!(pl1, ω_grid*0.5/pi, real(D2_tr.(im*ω_grid)), lab="Re{D2(ω)}")
#plot!(pl1, ω_grid*0.5/pi, real(D3_tr.(im*ω_grid)), lab="Re{D3(ω)}")
scatter!(pl1, [f_t], [0.0], markershape=:cross, label="Trmn(bare)")
scatter!(pl1, [f_t_corr], [1e-2], markershape=:cross, label="trmn(cpld)")
scatter!(pl1, [f_r], [0.0], markershape=:cross, label="Res(bare)")
scatter!(pl1, [f_r_corr], [1e-2], markershape=:cross, label="res(cpld)")

pl2 = plot(title = tstr, xaxis="freq [GHz]", yaxis=[-2,2], leg=:bottom)
plot!(pl2, ω_grid*0.5/pi, imag.(Z_tr.(im*ω_grid)), lab="Im(Z_tr(ω))")
scatter!(pl2, [f_t_corr], [0.0], markershape=:cross, label="Trmon(cpld)")
scatter!(pl2, [f_r_corr], [0.0], markershape=:cross, label="Reson(cpld)")
#plot!(pl2, [f_t_corr, f_t_corr], [1e-6, 1e1], ls=:dash, label="trmn(cpld)")
#plot!(pl2, [f_r_corr, f_r_corr], [1e-6, 1e1], ls=:dash, label="res(cpld)")


# Transfer fcn V_tr = T_ir * V_in (input to resonator)
Ck_o_Cr = 0.01
N_ir(s) = Ck_o_Cr*s^2*( ω_t^2 + s^2*(1 + Cg_o_Ct) )
D_ir(s) = D_tr(s) + Ck_o_Cr*s^2*(ω_t^2 + s^2*(1.0 + Cg_o_Ct))
T_ir(s) = (N_ir(s)/D_ir(s))
T_ri(s) = (D_ir(s)/N_ir(s)) # (res to input)

tstr = @sprintf("Cg/Ct = %.2f, Cg/Cr = %.2f, Ck/Cr = %.2f", Cg_o_Ct, Cg_o_Cr, Ck_o_Cr)
pl3 = plot(title = tstr, xaxis="freq [GHz]", leg=:bottom, yaxis=[-200, 100])
plot!(pl3, ω_grid*0.5/pi, real.(T_ri.(im*ω_grid)), lab="Re(T_ri(ω))")
scatter!(pl3, [f_t_corr], [0.0], markershape=:cross, label="Trmon(cpld)")
scatter!(pl3, [f_r_corr], [0.0], markershape=:cross, label="Reson(cpld)")

# Transfer fcn V_t = T_rt * V_tr (resonator to transmon)
N_rt(s) = s^2 * Cg_o_Ct
D_rt(s) = ω_t^2 + s^2*(1 + Cg_o_Ct)
T_rt(s) = N_rt(s)/D_rt(s)
T_tr(s) = D_rt(s)/N_rt(s) # (trans to res)
pl4 = plot(title = tstr, xaxis="freq [GHz]", leg=:bottom, yaxis=[-1, 10])
plot!(pl4, ω_grid*0.5/pi, real.(T_tr.(im*ω_grid)), lab="Re(T_tr(ω))")
scatter!(pl4, [f_t_corr], [0.0], markershape=:cross, label="Trmon(cpld)")
scatter!(pl4, [f_r_corr], [0.0], markershape=:cross, label="Reson(cpld)")

# Transfer fcn V_t = T_it * V_in (input to transmon)
T_it(s) = T_rt(s)*T_ir(s)
T_ti(s) = T_tr(s)*T_ri(s)
pl5 = plot(title = tstr, xlabel="freq [GHz]", leg=:topright, yaxis=[-150,250])
plot!(pl5, ω_grid*0.5/pi, real.(T_ti.(im*ω_grid)), lab="Re(T_ti(ω))")
scatter!(pl5, [f_t_corr], [0.0], markershape=:cross, label="Trmon(cpld)")
scatter!(pl5, [f_r_corr], [0.0], markershape=:cross, label="Reson(cpld)")

# Test T_it
N_it(s) = s^2*Cg_o_Ct * Z_tr(s) * Ck_o_Cr*s
D_it(s) = (ω_t^2 + s^2*(1+Cg_o_Ct)) * (1 + Ck_o_Cr*s*Z_tr(s))
T_it2(s) = N_it(s)/D_it(s)
T_ti2(s) = D_it(s)/N_it(s)
pl6 = plot(title = tstr, xlabel="freq [GHz]", leg=:topright, yaxis=[-150,250])
plot!(pl6, ω_grid*0.5/pi, real.(T_ti2.(im*ω_grid)), lab="Re(T_ti2(ω))")
scatter!(pl6, [f_t_corr], [0.0], markershape=:cross, label="Trmon(cpld)")
scatter!(pl6, [f_r_corr], [0.0], markershape=:cross, label="Reson(cpld)")


