using LinearAlgebra
using LinearAlgebra.BLAS
using MAT
using Test

path = dirname(@__FILE__)

ny = 3
ns = 10
np   = 2
nobs = 50

ws_l = KalmanLikelihoodWs{Float64, Integer}(ny, ns, np, nobs)

#=
y = randn(ny)
ystar = similar(y)
Z = randn(ny, ns)
Zstar = similar(Z)
H = randn(ny, ny)
H = H'*H
cholH = copy(H)
LAPACK.potrf!('L', H)
KalmanFilterTools.transformed_measurement!(ystar, Zstar, y, Z, cholH)
@test y ≈ LowerTriangular(cholH)*ystar
=#

nobs = 1
ws_l = KalmanLikelihoodWs{Float64, Integer}(ny, ns, np, nobs)
ws_f = KalmanFilterWs{Float64, Integer}(ny, ns, np, 3)
ws_d = DiffuseKalmanFilterWs{Float64, Integer}(ny, ns, np, 3)
Y = randn(ny, nobs+1)
t = 1
ystar = similar(Y)
Z = randn(ny, ns)
Zstar = similar(Z)
H = randn(ny, ny)
H = H'*H
#H = zeros(ny, ny)
copy!(ws_l.cholH, H)
LAPACK.potrf!('L', ws_l.cholH)
T = randn(ns, ns)
Q = randn(np, np)
Q = Q'*Q
R = randn(ns, np)
RQR = R*Q*R'
a0 = randn(ns)
P0 = randn(ns, ns)
P0 = P0'*P0
kalman_tol = eps()^(2/3)
diffuse_kalman_tol = kalman_tol

a = copy(a0)
aa0 = hcat(a0, zeros(ns), zeros(ns), zeros(ns))
P = copy(P0)
PP0 = cat(P0, zeros(ns,ns), zeros(ns,ns), zeros(ns,ns); dims=3)
ZZ = cat(Z[1,:]', Z[2,:]', Z[3,:]'; dims=3)
TT = I(ns) + zeros(ns, ns)
att = similar(aa0)
Ptt = similar(PP0)
lik1a = KalmanFilterTools.kalman_filter!(Y[:,1]', zeros(3), Z[1,:], H, zeros(ns), TT, R, zeros(np, np), aa0, att, PP0, Ptt, 1, 3, 0, ws_f,[[1], [1], [1]])


a = copy(a0)
P = copy(P0)
lik1 = KalmanFilterTools.kalman_likelihood(Y, Z, H, T, R, Q, a, P, 1, 1, 0, ws_l)
@test_broken aa0[:, 1] ≈ a
@test_broken PP0[:, :, 1] ≈ P
#@test lik0 ≈ ws_l.lik[1]  

a = copy(a0)
a00 = copy(a0)
P = copy(P0)
pattern = [1, 2, 3]
KalmanFilterTools.univariate_step!(Y, t, Z, diagm(diag(H)), T, RQR, a, P, kalman_tol, ws_f, pattern)

a = copy(a0)
P = copy(P0)
llik = 0.0
ndata = length(pattern)
i = 1
Zi = view(ws_f.Zstar, pattern[i], :)
v1 = KalmanFilterTools.get_v!(ws_f.ystar, ws_f.Zstar, a, pattern[i])
F = KalmanFilterTools.get_F(Zi, P, H[i,i], ws_f.PZi)
if abs(F) > kalman_tol
    a .+= (v1/F) .* ws_f.PZi
    # P = P - PZi*PZi'/F
    ger!(-1.0/F, ws_f.PZi, ws_f.PZi, P)
    llik += log(F) + v1*v1/F
end
Z1 = copy(Zi)
i = 2
Zi = view(ws_f.Zstar, pattern[i], :)
v2 = KalmanFilterTools.get_v!(ws_f.ystar, ws_f.Zstar, a, pattern[i])
F = KalmanFilterTools.get_F(Zi, P, H[i,i], ws_f.PZi)
if abs(F) > kalman_tol
    a .+= (v2/F) .* ws_f.PZi
    # P = P - PZi*PZi'/F
    ger!(-1.0/F, ws_f.PZi, ws_f.PZi, P)
    llik += log(F) + v2*v2/F
end
Z2 = Zi 

P = copy(PP0)
c = zeros(ny)
H = zeros(3, 3)
d = zeros(ns)
start = 1
data_pattern = [[1, 2, 3]]
changeC = ndims(c) > 1
changeH = ndims(H) > 2
changeD = ndims(d) > 1
changeT = ndims(T) > 2
changeR = ndims(R) > 2
changeQ = ndims(Q) > 2
changeA = ndims(a) > 1
changeP = ndims(P) > 2
changeK = ndims(ws_f.K) > 2
changeF = ndims(ws_f.F) > 2
changeiFv = ndims(ws_f.iFv) > 1
ny = size(Y, 1)
nobs = size(Y, 2) - start + 1
ns = size(T,1)
# QQ = R*Q*R'
vR = view(R, :, :, 1)
vQ = view(Q, :, :, 1)
KalmanFilterTools.get_QQ!(ws_f.QQ, vR, vQ, ws_f.RQ)
KalmanFilterTools.get_QQ!(ws_d.QQ, vR, vQ, ws_d.RQ)
l2pi = log(2*pi)
t = 1
steady = false
vP = view(P, :, :, 1)
copy!(ws_f.oldP, vP)
cholHset = false
pattern = data_pattern[t]
ndata = length(pattern)
vc = changeC ? view(c, :, t) : view(c, :)
#        ws_f.csmall .= view(vc, pattern)
vZsmall = KalmanFilterTools.get_vZsmall(ws_f.Zsmall, ws_f.iZsmall, Z, pattern, ndata, ny, t)
vH = changeH ? view(H, :, :, t) : view(H, :, :)
vT = changeT ? view(T, :, :, t) : view(T, :, :)
vR = changeR ? view(R, :, :, t) : view(R, :, :)
vQ = changeQ ? view(Q, :, :, t) : view(Q, :, :)
va = changeA ? view(a, :, t) : view(a, :)
vatt = changeA ? view(att, :, t) : view(att, :)
va1 = changeA ? view(a, :, t + 1) : view(a, :)
vd = changeD ? view(d, :, t) : view(d, :)
vP = changeP ? view(P, :, :, t) : view(P, :, :)
vPtt = changeP ? view(Ptt, :, :, t) : view(Ptt, :, :)
vP1 = changeP ? view(P, :, :, t + 1) : view(P, :, :)
vK = changeK ? view(ws_f.K, 1:ndata, :, t) : view(ws_f.K, 1:ndata, :)
if changeR || changeQ
    KalmanFilterTools.get_QQ!(ws_f.QQ, vR, vQ, ws_f.RQ)
end
viFv = changeiFv ? view(ws_f.iFv, 1:ndata, t) : view(ws_f.iFv, 1:ndata)
vv = view(ws_f.v, 1:ndata, t)
vF = changeF ? view(ws_f.F, 1:ndata, 1:ndata, t) : view(ws_f.F, 1:ndata, 1:ndata)
vvH = view(vH, pattern, pattern)
vZP = view(ws_f.ZP, 1:ndata, :)
vcholF = view(ws_f.cholF, 1:ndata, 1:ndata, t)
vcholH = view(ws_f.cholH, 1:ndata, 1:ndata)

# some observations this period
# v  = Y[:,t] - c - Z*a
KalmanFilterTools.get_v!(vv, Y, vc, vZsmall, va, t, pattern)
# F  = Z*P*Z' + H
KalmanFilterTools.get_F!(vF, vZP, vZsmall, vP, vvH)
info = KalmanFilterTools.get_cholF!(vcholF, vF)
info !=0 && error("F is near singular")

@testset "univariate_step" begin
    local a_u = similar(va) 
    local P_u = similar(vP)
    # multivariate treatment
    vatt_m = copy(vatt)
    va1_m  = copy(va1)
    vPtt_m = copy(vPtt)
    vP1_m  = copy(vP1)
    # iFv = inv(F)*v
    KalmanFilterTools.get_iFv!(viFv, vcholF, vv)
    lik_m = ndata*l2pi + log(KalmanFilterTools.det_from_cholesky(vcholF)) + LinearAlgebra.dot(vv, viFv)
    KalmanFilterTools.full_update!(va1_m, va, vatt_m, vd, vcholF, vK, vP1_m, vP, vPtt_m, vT, vv, vZP, d, steady, ws_f)
    
    # univariate treatment
    u_step = [
    () -> KalmanFilterTools.univariate_step!(Y, t, ws_f.Zsmall, vvH, T, ws_f.QQ, a_u, P_u, ws_f.kalman_tol, ws_f),
    () -> KalmanFilterTools.univariate_step!(Y, t, c, ws_f.Zsmall, vvH, d, T, ws_f.QQ, a_u, P_u, ws_f.kalman_tol, ws_f),
    () -> KalmanFilterTools.univariate_step!(Y, t, ws_f.Zsmall, vvH, T, ws_f.QQ, a_u, P_u, ws_f.kalman_tol, ws_f, pattern),
    () -> KalmanFilterTools.univariate_step!(Y, t, c, ws_f.Zsmall, vvH, d, T,ws_f.QQ, a_u, P_u, ws_f.kalman_tol, ws_f, pattern)        
    ]
    for us in u_step
        copy!(a_u, va)
        copy!(P_u, vP)
        lik_u = ndata*l2pi + us()
        @test a_u ≈ va1_m
        @test P_u ≈ vP1_m
        @test lik_u ≈ lik_m
    end
    local vatt_u = similar(va)
    local va1_u = similar(va)
    local vPtt_u = similar(vP)
    local vP1_u = similar(vP)
    u_step = [
    #       () -> KalmanFilterTools.univariate_step!(vatt_u, va1_u, vPtt_u, vP1_u, Y, t, ws_f.Zsmall, vvH, T, ws_f.QQ, a_u, P_u, ws_f.kalman_tol, ws_f)
    #       () -> KalmanFilterTools.univariate_step!(vatt_u, va1_u, vPtt_u, vP1_u, Y, t, c, ws_f.Zsmall, vvH, d, T, ws_f.QQ, a_u, P_u, ws_f.kalman_tol, ws_f)
    #       () -> KalmanFilterTools.univariate_step!(vatt_u, va1_u, vPtt_u, vP1_u, Y, t, ws_f.Zsmall, vvH, T, ws_f.QQ, a_u, P_u, ws_f.kalman_tol, ws_f, pattern)
    () -> KalmanFilterTools.extended_univariate_step!(vatt_u, va1_u, vPtt_u, vP1_u, Y, t, c, ws_f.Zsmall, vvH, d, T, ws_f.QQ, a_u, P_u, ws_f.kalman_tol, ws_f, pattern)
    ]
    for us in u_step
        copy!(a_u, va)
        copy!(P_u, vP)
        lik_u = ndata*l2pi + us()
        @test vatt_u ≈   vatt_m
        @test va1_u ≈ va1_m
        @test vPtt_u ≈ vPtt_m
        @test vP1_u ≈ vP1_m
        @test lik_u ≈ lik_m
    end
    local vPinf = copy(vP)
    local vPstar = copy(vP)
    local vPinftt_m = similar(vP)
    local vPstartt_m = similar(vP)
    local vPinf_u = similar(vP)
    local vPstar_u = similar(vP)
    vZPinf = vZsmall*vPinf
    vZPstar = vZsmall*vPstar
    vFstar = copy(vZPstar*vZsmall')
    vK0 = similar(ws_d.K)
    vK1 = similar(ws_d.K) 
    lik_m = ndata*l2pi + log(KalmanFilterTools.det_from_cholesky(vcholF)) + log(det(vFstar)) + vv'*inv(vFstar)*vv
    KalmanFilterTools.get_updated_Finfnonnull!(vatt_m,
                                            vPinftt_m,
                                            vPstartt_m,
                                            vZPinf,
                                            vZPstar,
                                            vcholF,    
                                            vFstar,    
                                            vZsmall,
                                            vPstar,
                                            vH,
                                            vK0,
                                            vK1,
                                            va,
                                            vv,
                                            vPinf,
                                            ws_d.PTmp)
    Minf = vPinf*vZsmall'
    Mstar = vPstar*vZsmall'
    Finf = vZsmall*Minf
    Fstar = vZsmall*Mstar + vH
    K0 = Minf*inv(Finf)
    K1 = Mstar*inv(Finf) - Minf*inv(Finf)*Fstar*inv(Finf)
    L0 = I - K0*vZsmall
    L1 = -K1*vZsmall 
    @test vPinftt_m ≈ vPinf*L0'                  
    @test vPstartt_m ≈ -vPinf*L1' + vPstar*L0'
    @test vatt_m ≈ va + K0*vv                      
    u_step = [
    () -> KalmanFilterTools.diffuse_univariate_step!(Y, t, vZsmall, vH, T, ws_d.QQ, a_u, vPinf_u, vPstar_u, diffuse_kalman_tol, kalman_tol, ws_d)
    () -> KalmanFilterTools.diffuse_univariate_step!(Y, t, vZsmall, vH, T, ws_d.QQ, a_u, vPinf_u, vPstar_u, diffuse_kalman_tol, kalman_tol, ws_d, pattern) 
    ]
    for (i, us) in enumerate(u_step)
        copy!(a_u, va)
        copy!(vPinf_u, vP)
        copy!(vPstar_u, vP)
        copy!(ws_d.v[:, 1], vv)
        lik_u = ndata*l2pi + us()
        @show i
        @test vPinf_u ≈ T*vPinftt_m*T'
        @test ws_d.QQ ≈ RQR
        @test vPstar_u ≈ T*vPstartt_m*T' + RQR
        @test a_u ≈ T*vatt_m
        @test lik_u ≈ lik_m
    end
    local vPinftt_u = similar(vP)
    local vPinf1_u = similar(vP)
    local vPstartt_u = similar(vP)
    local vPstar1_u = similar(vP)
    copy!(a_u, va)
    copy!(vPinf_u, vP)
    copy!(vPstar_u, vP)
    lik_u = ndata*l2pi + KalmanFilterTools.extended_diffuse_univariate_step!(vatt_u, va1_u, vPinftt_u, vPinf1_u, vPstartt_u, vPstar1_u, Y, t, c, Z, H, d, T, ws_d.QQ, a_u, vPinf_u, vPstar_u, diffuse_kalman_tol, kalman_tol, ws_d, pattern)
    # extended_diffuse_univariate_step!(att, a1, Pinftt, Pinf1, Pstartt, Pstar1,Y, t, c, Z, H, d, T, RQR, a, Pinf, Pstar, diffuse_klaman_tol, kalman_tol, ws, pattern)
    @test vatt_u ≈   vatt_m
    @test vPtt_u ≈ vPtt_m
    @test va1_u ≈ va1_m
    @test vP1_u ≈ vP1_m
    @test lik_u ≈ lik_m
end 

vars = matread("$path/reference/test_data.mat")

Y = vars["Y"]
Z = vars["Z"]
H = vars["H"]
T = vars["T"]
R = vars["R"]
Q = vars["Q"]
Pinf_0 = vars["Pinf"]
Pstar_0 = vars["Pstar"]

ny, nobs = size(Y)
ns, np = size(R)

a_0 = zeros(ns)
if H == 0
    H = zeros(ny, ny)
end

full_data_pattern = [collect(1:ny) for o = 1:nobs]

aa = zeros(ns, nobs + 1)
aa[:, 1] .= a_0
att = similar(aa)
Pinf = zeros(ns, ns, nobs + 1)
Pinftt = zeros(ns, ns, nobs + 1)
Pstar = zeros(ns, ns, nobs + 1)
Pstartt = zeros(ns, ns, nobs + 1)
Pinf[:, :, 1] = Pinf_0
Pinftt[:, :, 1] = Pinf_0
Pstar[:, :, 1] = Pstar_0
Pstartt[:, :, 1] =  Pstar_0
alphah = zeros(ns, nobs)
epsilonh = zeros(ny, nobs)
etah = zeros(np, nobs)
Valphah = zeros(ns, ns, nobs)
Vepsilonh = zeros(ny, ny, nobs)
Vetah = zeros(np, np, nobs)
c = zeros(ny)
d = zeros(ns)

ws6 = DiffuseKalmanSmootherWs(ny, ns, np, nobs)
llk_6b = diffuse_kalman_smoother!(Y, c, Z, H, d, T, R, Q, aa, att,
Pinf, Pinftt, Pstar, Pstartt,
alphah, epsilonh, etah, Valphah,
Vepsilonh, Vetah, 1, nobs, 0,
1e-8, ws6)

aa = zeros(ns, nobs + 1)
aa[:, 1] .= a_0
att = similar(aa)
Pinf = zeros(ns, ns, nobs + 1)
Pinftt = zeros(ns, ns, nobs + 1)
Pstar = zeros(ns, ns, nobs + 1)
Pstartt = zeros(ns, ns, nobs + 1)
Pinf[:, :, 1] = Pinf_0
Pinftt[:, :, 1] = Pinf_0
Pstar[:, :, 1] = Pstar_0
Pstartt[:, :, 1] =  Pstar_0
alphah = zeros(ns, nobs)
epsilonh = zeros(ny, nobs)
etah = zeros(np, nobs)
Valphah = zeros(ns, ns, nobs)
Vepsilonh = zeros(ny, ny, nobs)
Vetah = zeros(np, np, nobs)
c = zeros(ny)
d = zeros(ns)
r0 = randn(ns)
r0_1 = randn(ns)
r1 = randn(ns)
r1_1 = randn(ns)
L0 = Matrix{Float64}(undef, ns, ns)
L1 = similar(L0)
N0 = similar(L0)
N0_1 = similar(L0)
N1 = similar(L0)
N1_1 = similar(L0)
N2 = similar(L0)
N2_1 = similar(L0)
v = randn(ny)
ws_d = DiffuseKalmanSmootherWs(ny, ns, np, nobs)

y = Y[:, 1]
t = 1
a0 = copy(aa[:, 1])
a = copy(a0)
pinf0 = copy(Pinf[:, :, 1])
pinf = copy(pinf0)
pstar0 = copy(Pstar[:, :, 1])
pstar = copy(pstar0)
QQ = R*Q*R'
KalmanFilterTools.diffuse_univariate_step!(y, t, Z, H, T, QQ, a, pinf, pstar, 1e-10, 1e-10, ws_d)
v = y - c - Z*a0
K0 = T*pinf0*Z'*inv(Z*pinf0*Z')
a1 = T*a0 + K0*v
@test a ≈ a1

Finf0 = Z*pinf0*Z'
Finf = copy(Finf0)
Fstar0 = Z*pstar0*Z' + H
Fstar = copy(Fstar0)

K0 = pinf0*Z'*inv(Finf0)
K1 = pstar0*Z'*inv(Finf0) - pinf0*Z'*inv(Finf0)*Fstar0*inv(Finf0) 
K = copy(K0)

L0_target = I(ns) - K0*Z
L1_target = -K1*Z
r1_target = Z'inv(Finf)*v + L0_target'*r1 +L1_target'*r0
r0_target = L0_target'*r0
N0_target = L0_target'*N0*L0_target
N1_target = Z'*inv(Finf)*Z + L0_target'*N1*L0_target + L1_target'*N0_target*L0_target
Fstar = -inv(Finf)*Fstar*inv(Finf)
N2_target = Z'Fstar*Z + L0_target'*N2*L0_target' + L0_target'*N1_target*L1_target + L1_target*N1_target*L0_target + L1_target'*N0_target*L1_target 

tol = 1e-12
KalmanFilterTools.univariate_diffuse_smoother_step!(T, ws_d.F[:, :, 1], ws_d.Fstar[:, :, 1],
ws_d.K0[:, :, 1], ws_d.K[:, :, 1],
ws_d.L, ws_d.L1, ws_d.N, ws_d.N1,
ws_d.N2, r0, r1, ws_d.v[:,1], Z,
pinf, pstar, tol, ws_d)

#=
@test L0 ≈ L0_target
@test L1 ≈ L1_target
@test N0 ≈ N0_target
@test N1 ≈ N1_target
@test N2 ≈ N2_target
=#
@test r0 ≈ r0_target
@test r1 ≈ r1_target


