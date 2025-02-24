using LinearAlgebra
using Plots
using Printf
using Optim

# bare frequencies
f_t = 5.0 # GHz
f_r = 7.0 
# angular freq's
ω_t = 2*pi*f_t # rad / ns
ω_r = 2*pi*f_r

Cg_o_Ct = 0.1 # dimensionless
Cg_o_Cr = 0.2

Qr = 2.0e4 # 2.0e4 # quality factor of coil (=resonator?)
Rr_o_Lr = ω_r/Qr 
α_r = 0.5*Rr_o_Lr

Qt = 2.0e4 # 2.0e4 # quality factor of coil (=resonator?)
Rt_o_Lt = ω_t/Qt 
α_t = 0.5*Rt_o_Lt

# resonator with loss
Cr_Nr(s) = s + Rr_o_Lr # Cr * N_r
Dr(s) = s^2 + ω_r^2 + s*Rr_o_Lr
Cr_Zr(s) = (Cr_Nr(s)/Dr(s)) # Cr * Zr
Yr_o_Cr(s) = (Dr(s)/Cr_Nr(s))
ω_r_cpld = sqrt(ω_r^2 - Rr_o_Lr^2)
f_r_corr = ω_r_cpld*0.5/pi

# transmon with loss
Ct_Nt(s) = s + Rt_o_Lt # Ct * N_t
Dt(s) = s^2 + ω_t^2 + s*Rt_o_Lt
Ct_Zt(s) = (Ct_Nt(s)/Dt(s)) # Ct * Zt
Yt_o_Ct(s) = (Dt(s)/Ct_Nt(s))
ω_t_cpld = sqrt(ω_t^2 - Rt_o_Lt^2)
f_t_corr = ω_t_cpld*0.5/pi

# Transmon coupled to resonator
Ntr(s) =Cg_o_Cr * Cr_Zr(s)*(1/s + Cg_o_Ct*Ct_Zt(s))
Dtr(s) = 1/s + Cg_o_Cr*(Cr_Zr(s) + Cg_o_Ct*Ct_Zt(s))
Cg_Ztr(s) = Ntr(s) / Dtr(s) 
Ytr_o_Cg(s) = Dtr(s) / Ntr(s)

# define grid in 's'
Nω = 2000
ω_0 = 6.4 * 2*pi # (ω_t - 2*pi*2)
ω_1 = 6.5 * 2*pi # (ω_r + 2*pi*0.5)
ω_grid = LinRange(ω_0, ω_1, Nω)

tstr = @sprintf("f_r = %.3e, Rr_o_Lr = %.3e", f_r, Rr_o_Lr)
pl0 = plot(title = tstr, xaxis="freq [GHz]")
plot!(pl0, ω_grid*0.5/pi, real.(Cr_Nr.(im*ω_grid)), lab="Re{C_r*N_r(s)}")
plot!(pl0, ω_grid*0.5/pi, imag.(Cr_Nr.(im*ω_grid)), lab="Im{C_r*N_r(s)}")
plot!(pl0, ω_grid*0.5/pi, real.(Dr.(im*ω_grid)), lab="Re{D_r(s)}")
plot!(pl0, ω_grid*0.5/pi, imag.(Dr.(im*ω_grid)), lab="Im{D_r(s)}")
scatter!(pl0, [f_r], [0.0], markershape=:cross, label="Res(R=0)")
scatter!(pl0, [f_r_corr], [0.0], markershape=:cross, label="Res(R>0)")

# Impedance
tstr = @sprintf("Resonator impedance, Qr = %.2e", Qr)
pl1 = plot(title = tstr, xlabel="freq [GHz]", leg=:topright)
plot!(pl1, ω_grid*0.5/pi, real.(Cr_Zr.(im*ω_grid)), lab="Re(C_r*Zr(ω))")
plot!(pl1, ω_grid*0.5/pi, imag.(Cr_Zr.(im*ω_grid)), lab="Im(C_r*Zr(ω))")
plot!(pl1, ω_grid*0.5/pi, abs.(Cr_Zr.(im*ω_grid)), lab="Abs(C_r*Zr(ω))")

scatter!(pl1, [f_r], [0.0], markershape=:cross, label="Res(R=0)")
scatter!(pl1, [f_r_corr], [0.0], markershape=:cross, label="Res(R>0)")

# Admittence
tstr = @sprintf("Resonator admittence, Qr = %.2e", Qr)
pl2 = plot(title = tstr, xlabel="freq [GHz]", leg=:bottomright)
plot!(pl2, ω_grid*0.5/pi, real.(Yr_o_Cr.(im*ω_grid)), lab="Re(Yr/Cr(ω))")
plot!(pl2, ω_grid*0.5/pi, imag.(Yr_o_Cr.(im*ω_grid)), lab="Im(Yr/Cr(ω))")
plot!(pl2, ω_grid*0.5/pi, abs.(Yr_o_Cr.(im*ω_grid)), lab="Abs(Yr/Cr(ω))")

scatter!(pl2, [f_r], [0.0], markershape=:cross, label="Res(R=0)")
scatter!(pl2, [f_r_corr], [0.0], markershape=:cross, label="Res(R>0)")

# define grid in 's' near omega_r
Nω = 2000
ω_0 = 6.4 * 2*pi # (ω_t - 2*pi*2)
ω_1 = 6.43 * 2*pi # (ω_r + 2*pi*0.5)
ω_grid = LinRange(ω_0, ω_1, Nω)

rad2deg = 180/pi
# Impedance
tstr = @sprintf("Transmon+Resonator impedance near ω_r ")
pl3 = plot(title = tstr, xlabel="freq [GHz]", leg=:topright)
plot!(pl3, ω_grid*0.5/pi, real.(Cg_Ztr.(im*ω_grid)), lab="Re(C_g*Ztr(ω))")
plot!(pl3, ω_grid*0.5/pi, imag.(Cg_Ztr.(im*ω_grid)), lab="Im(C_g*Ztr(ω))")
#plot!(pl3, ω_grid*0.5/pi, rad2deg*angle.(Cg_Ztr.(im*ω_grid)), lab="phase(C_g*Ztr(ω))")

# define grid in 's'
Nω = 2000
ω_0 = 4.925 * 2*pi # (ω_t - 2*pi*2)
ω_1 = 4.95 * 2*pi # (ω_r + 2*pi*0.5)
ω_grid = LinRange(ω_0, ω_1, Nω)

# Impedance
tstr = @sprintf("Transmon+Resonator impedance near ω_t")
pl4 = plot(title = tstr, xlabel="freq [GHz]", leg=:topright)
plot!(pl4, ω_grid*0.5/pi, real.(Cg_Ztr.(im*ω_grid)), lab="Re(C_g*Ztr(ω))")
plot!(pl4, ω_grid*0.5/pi, imag.(Cg_Ztr.(im*ω_grid)), lab="Im(C_g*Ztr(ω))")
#plot!(pl4, ω_grid*0.5/pi, abs.(Cg_Ztr.(im*ω_grid)), lab="Abs(C_g*Ztr(ω))")

# Transfer fcn V_tr = T_itr * V_in (input to resonator)
Ck_o_Cg = 0.01
N_rt(s) = Ck_o_Cg * Cg_Ztr(s)
D_rt(s) = 1/s + Ck_o_Cg * Cg_Ztr(s)
T_itr(s) = N_rt(s)/D_rt(s) # (in to res)
T_tri(s) = D_rt(s)/N_rt(s) # (res to in)

# define grid in 's' near omega_r
Nω = 2000
ω_0 = 6.40 * 2*pi # (ω_t - 2*pi*2)
ω_1 = 6.42 * 2*pi # (ω_r + 2*pi*0.5)
ω_grid = LinRange(ω_0, ω_1, Nω) 

tstr = @sprintf("Transfer func res to input, near res freq")
pl5 = plot(title = tstr, xlabel="freq [GHz]", ylabel="Phase [deg]", leg=:bottomright)
#plot!(pl5, ω_grid*0.5/pi, real.(T_tri.(im*ω_grid)), lab="Re(T_tri(ω))")
#plot!(pl5, ω_grid*0.5/pi, imag.(T_tri.(im*ω_grid)), lab="Im(T_tri(ω))")
plot!(pl5, ω_grid*0.5/pi, rad2deg*angle.(T_tri.(im*ω_grid)), lab="phase(T_tri(ω))")

tstr = @sprintf("Transfer func input to res, near res freq")
pl5n = plot(title = tstr, xlabel="freq [GHz]", ylabel="Phase [deg]", leg=:bottomright)
plot!(pl5n, ω_grid*0.5/pi, rad2deg*angle.(T_itr.(im*ω_grid)), lab="phase(T_itr(ω))")

# define grid in 's' near omega_t
Nω = 2000
ω_0 = 4.93 * 2*pi # (ω_t - 2*pi*2)
ω_1 = 4.95 * 2*pi # (ω_r + 2*pi*0.5)
ω_grid = LinRange(ω_0, ω_1, Nω) 

tstr = @sprintf("Transfer func res to input, near transmon freq")
pl6 = plot(title = tstr, xaxis="freq [GHz]", ylabel="Phase [deg]", leg=:bottomright)
#plot!(pl6, ω_grid*0.5/pi, real.(T_tri.(im*ω_grid)), lab="Re(T_tri(ω))")
#plot!(pl6, ω_grid*0.5/pi, imag.(T_tri.(im*ω_grid)), lab="Im(T_tri(ω))")
plot!(pl6, ω_grid*0.5/pi, rad2deg*angle.(T_tri.(im*ω_grid)), lab="phase(T_tri(ω))")

tstr = @sprintf("Transfer func res to input, near transmon freq")
pl6n = plot(title = tstr, xaxis="freq [GHz]", ylabel="Phase [deg]", leg=:right)
plot!(pl6n, ω_grid*0.5/pi, rad2deg*angle.(T_tri.(im*ω_grid)), lab="phase(T_itr(ω))")

# define grid in 's' near omega_t
Nω = 2000
ω_0 = 4.93 * 2*pi # (ω_t - 2*pi*2)
ω_1 = 4.95 * 2*pi # (ω_r + 2*pi*0.5)
ω_grid = LinRange(ω_0, ω_1, Nω)
pl6a = plot(title = tstr, xaxis="freq [GHz]", ylabel="Magnitude", leg=:right)
plot!(pl6a, ω_grid*0.5/pi, abs.(T_tri.(im*ω_grid)), lab="abs(T_itr(ω))")
#scatter!(pl6, [f_t_corr], [0.0], markershape=:cross, label="Trmon(cpld)")
#scatter!(pl6, [f_r_corr], [0.0], markershape=:cross, label="Reson(cpld)")

# Transfer fcn V_t = T_trt * V_tr (resonator to transmon)
N_trt(s) = Ct_Zt(s)
D_trt(s) = Cg_o_Ct *1/s + Ct_Zt(s)
T_trt(s) = N_trt(s)/D_trt(s) # (in to res)

# define grid in 's' full domain
Nω = 2000
ω_0 = 4.0 * 2*pi # (ω_t - 2*pi*2)
ω_1 = 7.5 * 2*pi # (ω_r + 2*pi*0.5)
ω_grid = LinRange(ω_0, ω_1, Nω) 

tstr = @sprintf("Transfer func res to transmon")
pl7 = plot(title = tstr, xaxis="freq [GHz]", leg=:right)
plot!(pl7, ω_grid*0.5/pi, real.(T_trt.(im*ω_grid)), lab="Re(T_trt(ω))")
plot!(pl7, ω_grid*0.5/pi, imag.(T_trt.(im*ω_grid)), lab="Im(T_trt(ω))")
plot!(pl7, ω_grid*0.5/pi, abs.(T_trt.(im*ω_grid)), lab="Abs(T_trt(ω))")
