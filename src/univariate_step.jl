#=
function univariate_step!(t, Y, Z, H, T, QQ, a, P, kalman_tol, ws)
    ws.vH = changeH ? view(H, :, :, t) : view(H, :, :)
    if isdiag(vH)
        univariate_step_0(y, Z, vH, T, QQ, a, P, kalman_tol, ws)
    else
        copy!(ws.ystar, y)
        transformed_measurement!(ws.ystar, ws.Zstar, ws.Hstar, y, Z, ws.vH, changeH)
        univariate_step_0(ws.ystar, ws.Zstar, ws.Hstar, T, QQ, a, P, kalman_tol, ws)
    end
end
=#

#=
function transformed_measurement!(ystar, Zstar, y, Z, cholH)
    UTcholH = UpperTriangular(cholH)
    ystar .= transpose(UTcholH)\y
    Zstar .= transpose(UTcholH)\Z
    #=`
    copy!(ystar, y)
    ldiv!(UTcholH, ystar)
    copy!(Zstar, Z)
    ldiv!(UTcholH, Zstar)
    =#
end
=#

function logproddiag(A)
    @assert isdiag(A)
    y = 0
    for i in 1:size(A, 1)
        y += log(A[i, i])
    end
    return y
end

function univariate_step!(Y, t, Z, H, T, RQR, a, P, kalman_tol, ws)
    ny = size(Y,1)
    if !isdiag(H)
        error("singular F with non-diagonal H matrix not yet supported")
        transformed_measurement!(ws.ystar, ws.Zstar, view(Y, :, t), Z, ws.cholH)
        H = I(ny)
	logdetcholH = 0.0
    else
        copy!(ws.ystar, view(Y, :, t))
        copy!(ws.Zstar, Z)
        logdetcholH = logproddiag(H)
    end
    llik = 0.0
    for i=1:ny
        Zi = view(ws.Zstar, i, :)
        v = get_v!(ws.ystar, ws.Zstar, a, i)
        F = get_F(Zi, P, H[i,i], ws.PZi)
        if abs(F) > kalman_tol
            a .+= (v/F) .* ws.PZi
            # P = P - PZi*PZi'/F
            ger!(-1.0/F, ws.PZi, ws.PZi, P)
            llik += log(F) + v*v/F
        end
    end
    mul!(ws.a1, T, a)
    a .= ws.a1
    mul!(ws.PTmp, T, P)
    copy!(P, RQR)
    mul!(P, ws.PTmp, T', 1.0, 1.0)
    return llik + 2*logdetcholH
end

function univariate_step!(Y, t, Z, H, T, RQR, a, P, kalman_tol, ws, pattern)
    ny = size(Y,1)
    if !isdiag(H)
        error("singular F with non-diagonal H matrix not yet supported")
        transformed_measurement!(ws.ystar, ws.Zstar, view(Y, :, t), Z, ws.cholH)
        H = I(ny)
	logdetcholH = 0.0
    else
        copy!(ws.ystar, view(Y, :, t))
        copy!(ws.Zstar, Z)
        logdetcholH = logproddiag(H)
    end
    llik = 0.0
    ndata = length(pattern)
    for i in 1:ndata
        Zi = view(ws.Zstar, pattern[i], :)
        v = get_v!(ws.ystar, ws.Zstar, a, pattern[i])
        F = get_F(Zi, P, H[i,i], ws.PZi)
        if abs(F) > kalman_tol
            a .+= (v/F) .* ws.PZi
            # P = P - PZi*PZi'/F
            ger!(-1.0/F, ws.PZi, ws.PZi, P)
            llik += log(F) + v*v/F
        end
    end
    mul!(ws.a1, T, a)
    a .= ws.a1
    mul!(ws.PTmp, T, P)
    copy!(P, RQR)
    mul!(P, ws.PTmp, T', 1.0, 1.0)
    return llik + 2*logdetcholH
end

function univariate_step(Y, t, Z, H, T, QQ, a, Pinf, Pstar, diffuse_kalman_tol, kalman_tol, ws)
    ny = size(Y,1)
    if !isdiag(H)
        error("singular F with non-diagonal H matrix not yet supported")
        transformed_measurement!(ws.ystar, ws.Zstar, view(Y, :, t), Z, ws.cholH)
        H = I(ny)
	logdetcholH = 0.0
    else
        copy!(ws.ystar, view(Y, :, t))
        copy!(ws.Zstar, Z)
        logdetcholH = logproddiag(H)
    end
    llik = 0.0
    for i=1:size(Y,1)
        Zi = view(Z, i, :)
        v = get_v(ws.ystar, ws.Zstar, a, i)
        Fstar = get_Fstar!(Zi, Pstar, H[i], ws.uK1)
        Finf = get_Finf!(Zi, Pstar, ws.uK1)
        # Conduct check of rank
        # Pinf and Finf are always scaled such that their norm=1: Fstar/Pstar, instead,
        # depends on the actual values of std errors in the model and can be badly scaled.
        # experience is that diffuse_kalman_tol has to be bigger than kalman_tol, to ensure
        # exiting the diffuse filter properly, avoiding tests that provide false non-zero rank for Pinf.
        # Also the test for singularity is better set coarser for Finf than for Fstar for the same reason
        if Finf > diffuse_kalman_tol                 # F_{\infty,t,i} = 0, use upper part of bracket on p. 175 DK (2012) for w_{t,i}
            ws.K0_Finf .= ws.uK0./Finf
            a .+= ws.v[i].*ws.K0_Finf
            # Pstar     = Pstar + K0*(K0_Finf'*(Fstar/Finf)) - K1*K0_Finf' - K0_Finf*K1'
            ger!( Fstar/Finf, ws.uK0, ws.K0_Finf, Pstar)
            ger!( -1.0, ws.uK1, ws.K0_Finf, Pstar)
            ger!( -1.0, ws.K0_Finf, ws.uK1, Pstar)
            # Pinf      = Pinf - K0*K0_Finf'
            ger!(-1.0, ws.uK0, ws.K0_Finf, Pinf)
            llik += log(Finf)
        elseif Fstar > kalman_tol
            llik += log(Fstar) + ws.v[i]*ws.v[i]/Fstar
            a .+= ws.uK1.*(ws.v[i]/Fstar)
            ger!(-1/Fstar, ws.uK1, ws.uK1, Pstar)
        else
            # do nothing as a_{t,i+1}=a_{t,i} and P_{t,i+1}=P_{t,i}, see
            # p. 157, DK (2012)
        end
    end
    mul!(ws.a1, T, a)
    a .= ws.a1
    mul!(ws.PTmp, T, P)
    copy!(P, RQR)
    mul!(P, ws.PTmp, T', 1.0, 1.0)
    return llik + 2*logdetcholH
end

function univariate_step(Y, t, Z, H, T, QQ, a, Pinf, Pstar, diffuse_kalman_tol, kalman_tol, ws, pattern)
    ny = size(Y,1)
    if !isdiag(H)
        error("singular F with non-diagonal H matrix not yet supported")
        transformed_measurement!(ws.ystar, ws.Zstar, view(Y, :, t), Z, ws.cholH)
        H = I(ny)
	logdetcholH = 0.0
    else
        copy!(ws.ystar, view(Y, :, t))
        copy!(ws.Zstar, Z)
        logdetcholH = logproddiag(H)
    end
    llik = 0.0
    ndata = length(pattern)
    for i=1:ndata
        Zi = view(ws.Zstar, pattern[i], :)
        v = get_v(ws.ystar, ws.Zstar, a, pattern[i])
        Fstar = get_Fstar!(Zi, Pstar, H[i], ws.uK1)
        Finf = get_Finf!(Zi, Pstar, ws.uK1)
        # Conduct check of rank
        # Pinf and Finf are always scaled such that their norm=1: Fstar/Pstar, instead,
        # depends on the actual values of std errors in the model and can be badly scaled.
        # experience is that diffuse_kalman_tol has to be bigger than kalman_tol, to ensure
        # exiting the diffuse filter properly, avoiding tests that provide false non-zero rank for Pinf.
        # Also the test for singularity is better set coarser for Finf than for Fstar for the same reason
        if Finf > diffuse_kalman_tol                 # F_{\infty,t,i} = 0, use upper part of bracket on p. 175 DK (2012) for w_{t,i}
            ws.K0_Finf .= ws.uK0./Finf
            a .+= prediction_error.*ws.K0_Finf
            # Pstar     = Pstar + K0*(K0_Finf'*(Fstar/Finf)) - K1*K0_Finf' - K0_Finf*K1'
            ger!( Fstar/Finf, ws.uK0, ws.K0_Finf, Pstar)
            ger!( -1.0, ws.uK1, ws.K0_Finf, Pstar)
            ger!( -1.0, ws.K0_Finf, ws.uK1, Pstar)
            # Pinf      = Pinf - K0*K0_Finf'
            ger!(-1.0, ws.uK0, ws.K0_Finf, Pinf)
            llik += log(Finf)
        elseif Fstar > kalman_tol
            llik += ndata*l2pi, log(Fstar) + v*v/Fstar
            a .+= ws.uK1.*(v/Fstar)
            ger!(-1/Fstar, ws.uK1, ws.uK1, Pstar)
        else
            # do nothing as a_{t,i+1}=a_{t,i} and P_{t,i+1}=P_{t,i}, see
            # p. 157, DK (2012)
        end
    end
    mul!(ws.a1, T, a)
    a .= ws.a1
    mul!(ws.PTmp, T, P)
    copy!(P, RQR)
    mul!(P, ws.PTmp, T', 1.0, 1.0)
    return llik + 2*logdetcholH
end

function univariate_step!(Y, c, t, Z, H, d, T, RQR, a, P, kalman_tol, ws)
    ny = size(Y,1)
    if !isdiag(H)
        error("singular F with non-diagonal H matrix not yet supported")
        transformed_measurement!(ws.ystar, ws.Zstar, view(Y, :, t), Z, ws.cholH)
        H = I(ny)
	logdetcholH = 0.0
    else
        copy!(ws.ystar, view(Y, :, t))
        copy!(ws.Zstar, Z)
        logdetcholH = logproddiag(H)
    end
    llik = 0.0
    for i=1:ny
        Zi = view(ws.Zstar, i, :)
        v = get_v!(ws.ystar, c, ws.Zstar, a, i)
        F = get_F(Zi, P, H[i,i], ws.PZi)
        if abs(F) > kalman_tol
            a .+= d[i] + (v/F) .* ws.PZi
            # P = P - PZi*PZi'/F
            ger!(-1.0/F, ws.PZi, ws.PZi, P)
            llik += log(F) + v*v/F
        end
    end
    copy!(ws.a1, d)
    mul!(ws.a1, T, a, 1.0, 1.0)
    a .= ws.a1
    mul!(ws.PTmp, T, P)
    copy!(P, RQR)
    mul!(P, ws.PTmp, T', 1.0, 1.0)
    return llik + 2*logdetcholH
end


function univariate_step!(att, a1, Ptt, P1, Y, t, c, Z, H, d, T, RQR, a, P, kalman_tol, ws, pattern)
    ny = size(Y,1)
    if !isdiag(H)
        error("singular F with non-diagonal H matrix not yet supported")
        transformed_measurement!(ws.ystar, ws.Zstar, view(Y, :, t), Z, ws.cholH)
        H = I(ny)
	logdetcholH = 0.0
    else
        copy!(ws.ystar, view(Y, :, t))
        copy!(ws.Zstar, Z)
        logdetcholH = logproddiag(H)
    end
    llik = 0.0
    ndata = length(pattern)
    copy!(att, a)
    copy!(Ptt, P)
    for i=1:ndata
        Zi = view(ws.Zstar, pattern[i], :)
        v = get_v!(ws.ystar, c, ws.Zstar, att, pattern[i])
        F = get_F(Zi, Ptt, H[i,i], ws.PZi)
        if abs(F) > kalman_tol
            att .+= (v/F) .* ws.PZi
            # P = P - PZi*PZi'/F
            ger!(-1.0/F, ws.PZi, ws.PZi, Ptt)
            llik += log(F) + v*v/F
        end
    end
    copy!(a1, d)
    mul!(a1, T, att, 1.0, 1.0)
    mul!(ws.PTmp, T, Ptt)
    copy!(P1, RQR)
    mul!(P1, ws.PTmp, T', 1.0, 1.0)
    return llik + 2*logproddiag(H)
end

function univariate_step(Y, t, c, Z, H, d, T, RQR, a, Pinf, Pstar, diffuse_kalman_tol, kalman_tol, ws)
    ny = size(Y,1)
    if !isdiag(H)
        error("singular F with non-diagonal H matrix not yet supported")
        transformed_measurement!(ws.ystar, ws.Zstar, view(Y, :, t), Z, ws.cholH)
        H = I(ny)
	logdetcholH = 0.0
    else
        copy!(ws.ystar, view(Y, :, t))
        copy!(ws.Zstar, Z)
        logdetcholH = logproddiag(H)
    end
    llik = 0.0
    for i=1:size(Y,1)
        Zi = view(Z, i, :)
        uK0 = view(ws.K0, i, :, t)
        uK1 = view(ws.K, i, :, t)
        ws.v[i] = get_v!(ws.ystar, c, ws.Zstar, a, i)
        Fstar = get_Fstar!(Zi, Pstar, H[i], uK1)
        Finf = get_Finf!(Zi, Pinf, uK0)
        # Conduct check of rank
        # Pinf and Finf are always scaled such that their norm=1: Fstar/Pstar, instead,
        # depends on the actual values of std errors in the model and can be badly scaled.
        # experience is that diffuse_kalman_tol has to be bigger than kalman_tol, to ensure
        # exiting the diffuse filter properly, avoiding tests that provide false non-zero rank for Pinf.
        # Also the test for singularity is better set coarser for Finf than for Fstar for the same reason
        if Finf > diffuse_kalman_tol                 # F_{\infty,t,i} = 0, use upper part of bracket on p. 175 DK (2012) for w_{t,i}
            ws.K0_Finf .= uK0./Finf
            a .+= ws.v[i]*ws.K0_Finf
            # Pstar     = Pstar + K0*(K0_Finf'*(Fstar/Finf)) - K1*K0_Finf' - K0_Finf*K1'
            ger!( Fstar/Finf, uK0, ws.K0_Finf, Pstar)
            ger!( -1.0, uK1, ws.K0_Finf, Pstar)
            ger!( -1.0, ws.K0_Finf, uK1, Pstar)
            # Pinf      = Pinf - K0*K0_Finf'
            ger!(-1.0, uK0, ws.K0_Finf, Pinf)
            llik += log(Finf)
        elseif Fstar > kalman_tol
            llik += log(Fstar) + ws.v[i]*ws.v[i]/Fstar
            a .+= uK1.*(ws.v[i]/Fstar)
            ger!(-1/Fstar, uK1, uK1, Pstar)
        else
            # do nothing as a_{t,i+1}=a_{t,i} and P_{t,i+1}=P_{t,i}, see
            # p. 157, DK (2012)
        end
        ws.F[i, i, t] = Finf
        ws.Fstar[i, i, t] = Fstar
    end
    copy!(ws.a1, d)
    mul!(ws.a1, T, a, 1.0, 1.0)
    a .= ws.a1
    mul!(ws.PTmp, T, Pinf)
    mul!(Pinf, ws.PTmp, T')
    mul!(ws.PTmp, T, Pstar)
    copy!(Pstar, RQR)
    mul!(Pstar, ws.PTmp, T', 1.0, 1.0)
    return llik + 2*logdetcholH
end

function univariate_step(att, a1, Pinftt, Pinf1, Pstartt, Pstar1, Y, t, c, Z, H, d, T, QQ, a, Pinf, Pstar, diffuse_kalman_tol, kalman_tol, ws, pattern)
    copy!(a1, a)
    copy!(Pinftt, Pinf)
    copy!(Pstartt, Pstar)
    ny = size(Y,1)
    ndata = length(pattern)
    vystar = view(ws.ystar, 1:ndata)
    vZstar = view(ws.Zstar, 1:ndata, :)
    if !isdiag(H)
        error("singular F with non-diagonal H matrix not yet supported")
        transformed_measurement!(ws.ystar, vZstar, view(Y, :, t), Z, ws.cholH)
        H = I(ny)
	logdetcholH = 0.0
    else
        copy!(vystar, view(Y, pattern, t))
        copy!(vZstar, Z)
        logdetcholH = logproddiag(H)
    end
    llik = 0.0
    for i=1:ndata
        uK0 = view(ws.K0, i, :, t)
        uK1 = view(ws.K, i, :, t)
        Zi = view(vZstar, i, :)
        ws.v[i] = get_v!(vystar, c, vZstar, a, i)
        Fstar = get_Fstar!(Zi, Pstartt, H[i], uK1)
        Finf = get_Finf!(Zi, Pinftt, uK0)
        # Conduct check of rank
        # Pinf and Finf are always scaled such that their norm=1: Fstar/Pstar, instead,
        # depends on the actual values of std errors in the model and can be badly scaled.
        # experience is that diffuse_kalman_tol has to be bigger than kalman_tol, to ensure
        # exiting the diffuse filter properly, avoiding tests that provide false non-zero rank for Pinf.
        # Also the test for singularity is better set coarser for Finf than for Fstar for the same reason
        if Finf > diffuse_kalman_tol                 # F_{\infty,t,i} = 0, use upper part of bracket on p. 175 DK (2012) for w_{t,i}
            ws.K0_Finf .= uK0./Finf
            att .+= ws.v[i] .* ws.K0_Finf
            # Pstar     = Pstar + K0*(K0_Finf'*(Fstar/Finf)) - K1*K0_Finf' - K0_Finf*K1'
            ger!( Fstar/Finf, uK0, ws.K0_Finf, Pstartt)
            ger!( -1.0, uK1, ws.K0_Finf, Pstartt)
            ger!( -1.0, ws.K0_Finf, uK1, Pstartt)
            # Pinf      = Pinf - K0*K0_Finf'
            ger!(-1.0, uK0, ws.K0_Finf, Pinftt)
            llik += log(Finf)
        elseif Fstar > kalman_tol
            llik += log(Fstar) + ws.v[i]*ws.v[i]/Fstar
            att .+= uK1.*(ws.v[i]/Fstar)
            ger!(-1/Fstar, uK1, uK1, Pstartt)
        else
            # do nothing as a_{t,i+1}=a_{t,i} and P_{t,i+1}=P_{t,i}, see
            # p. 157, DK (2012)
        end
        ws.F[i, i, t] = Finf
        ws.Fstar[i, i, t] = Fstar
    end
    copy!(a1, d)
    mul!(a1, T, att, 1.0, 1.0)
    mul!(ws.PTmp, T, Pinftt)
    mul!(Pinf1, ws.PTmp, transpose(T))
    mul!(ws.PTmp, T, Pstartt)
    copy!(Pstar1, QQ)
    mul!(Pstar1, ws.PTmp, transpose(T), 1.0, 1.0)
    return llik + 2*logdetcholH
end

function univariate_diffuse_smoother_step!(T::AbstractMatrix{W},
                                           Finf::AbstractMatrix{W},
                                           Fstar::AbstractMatrix{W},
                                           K0::AbstractMatrix{W},
                                           K1::AbstractMatrix{W},
                                           L0::AbstractMatrix{W},
                                           L1::AbstractMatrix{W},
                                           N0::AbstractMatrix{W},
                                           N1::AbstractMatrix{W},
                                           N2::AbstractMatrix{W},
                                           r0::AbstractVector{W},
                                           r1::AbstractVector{W},
                                           v::AbstractVector{W},
                                           Z::AbstractArray{U},
                                           tol::W,
                                           ws::DiffuseKalmanSmootherWs) where {W <: AbstractFloat,
                                                                               U <: Real}
    ny = size(Finf, 1)
    ns = size(L0, 1)
    for i = ny: -1: 1
        if Finf[i, i] > tol
            iFinf = 1/Finf[i, i]
            vK0 = view(K0, i, :)
            vK1 = view(K1, i, :)
            # get_L0!(L0, K0, Z, Finf, i) 
            copy!(L0, I(ns))
            vZ = view(Z, i, :)
            ger!(-iFinf, vK0, vZ, L0)
            # get_L1!(L1, K0, K1, Finf, Fstar, Z, i)  
            fill!(L1, 0.0)
            lmul!(Fstar[i, i]/Finf[i, i], vK0)
            vK0 .-= vK1
            ger!(iFinf, vK0, vZ, L1)
            # compute r1_{t,i-1} first because it depends
            # upon r0{t,i} 
            # update_r1!(r1, r0, Z, v, Finv, L0, L1, i)
            copy!(ws.tmp_ns, vZ)
            rmul!(ws.tmp_ns, v[i]*iFinf)
            mul!(ws.tmp_ns, transpose(L1), r0, 1.0, 1.0)
            mul!(ws.tmp_ns, transpose(L0), r1, 1.0, 1.0)
            copy!(r1, ws.tmp_ns)
            #  r0(:,t) = Linf'*r0(:,t);   % KD (2000), eq. (25) for r_0
            copy!(ws.tmp_ns, r0)
            mul!(r0, transpose(L0), ws.tmp_ns)
            # update_N2!(N2, viFZ, vFstar, L0, N2_1, N1_1,
            #            L1, N0_1, vTmp, ws.PTmp)
            mul!(ws.PTmp, transpose(L0), N2)
            mul!(ws.PTmp, transpose(L1), N1, 1.0, 1.0)
            mul!(N2, ws.PTmp, L0)
            mul!(ws.PTmp, transpose(L0), N1)
            mul!(ws.PTmp, transpose(L1), N0, 1.0, 1.0)
            mul!(N2, ws.PTmp, L1, 1.0, 1.0)
            ger!(Fstar[i]/(Finf[i]*Finf[i]),vZ, vZ, N2)
            # update_N1!(N1, vZ, Finf, L0, N1, L1, N0, ws.PTmp)
            mul!(ws.PTmp, transpose(L0), N1)
            mul!(ws.PTmp, transpose(L1), N0, 1.0, 1.0)
            mul!(N1, ws.PTmp, L0)
            ger!(iFinf, vZ, vZ, N1)
            # update_N0!(N0, L1, ws.PTmp)
            mul!(ws.PTmp, transpose(L0), N0)
            mul!(N0, ws.PTmp, L0)
        elseif Fstar[i, i] > tol
            iFstar = 1/Fstar[i, i]
            # get_L0!(L1, K1, Z, Fstar, i)
            copy!(L0, I(ns))
            vK1 = view(K1, i, :)
            vZ = view(Z, i, :)
            ger!(-iFstar, vK1, vZ, L0)
            # update_r0!(r0, Z, Fstar, v, L0, i)
            copy!(ws.tmp_ns, r0)
            r0 .= vZ
            rmul!(r0, v[i]*iFstar)
            mul!(r0, transpose(L0), ws.tmp_ns, 1.0, 1.0)
            # update_N0!(N0, L1, ws.PTmp)
            mul!(ws.PTmp, transpose(L0), N0)
            mul!(N0, ws.PTmp, L0)
            ger!(iFstar, vZ, vZ, N0)
        end
    end
    #=
    =#
end
            
function univariate_diffuse_smoother_step!(T::AbstractMatrix{W},
                                           Finf::AbstractMatrix{W},
                                           Fstar::AbstractMatrix{W},
                                           K0::AbstractMatrix{W},
                                           K1::AbstractMatrix{W},
                                           L0::AbstractMatrix{W},
                                           L1::AbstractMatrix{W},
                                           r0::AbstractVector{W},
                                           r1::AbstractVector{W},
                                           v::AbstractVector{W},
                                           Z::AbstractArray{U},
                                           tol::W,
                                           ws::DiffuseKalmanSmootherWs) where {W <: AbstractFloat,
                                                                               U <: Real}
    ny = size(Finf, 1)
    ns = size(L0, 1)
    for i = 1: ny
        if Finf[i, i] > tol
            iFinf = 1/Finf[i,i]
            # get_L1!(L1, K0, Z, Finf, i) 
            copy!(L1,I(ns))
            vK0 = view(K0, :, i)
            vZ = view(Z, i, :)
            ger!(-iFinf, vK0, vZ, L1)
            # get_L0!(L0, K0, K1, Finf, Fstar, Z, i)  
            fill!(L0, 0.0)
            vFstar = view(Fstar, i, :)
            lmul!(Fstar[i]*Finf, vK0)
            vK0 .-= vK1
            ger!(1.0, vK1, vZ, L0) 
            # update_r1!(r1, r0, Z, v, Finv, L0, L1, i)
            r1 .= vZ
            rmul!(r1, v[i]*iFinf)
            mul!(r1, transpose(L0), r0, 1.0, 1.0)
            mul!(r1, transpose(L1), r1, 1.0, 1.0)
            #  r0(:,t) = Linf'*r0(:,t);   % KD (2000), eq. (25) for r_0
            copy!(ws.tmp_ns, r0)
            mul!(r0, transpose(L0), ws.tmp_ns)
        elseif Fstar[i, i] > tol
            iFstar = 1/Fstar[i,i]
            # get_L1!(L1, K1, Z, Fstar, i)
            copy!(L1, I(ny))
            vK1 = view(K1, :, i)
            vZ = view(Z, i, :)
            ger!(-iFstar, vK1, vZ)
            # update_r0!(r0, Z, Fstar, v, L1, i)
            r0 .= vZ
            rmul!(r0, v[i]*iFstar)
            mul!(r0, transpose(L1), r0, 1.0, 1.0)
        end
    end
    #=
    copy!(ws.tmp_ns, r0)
    mul!(r0, transpose(T), ws.tmp_ns)
    copy!(ws.tmp_ns, r1)
    mul!(r1, transpose(T), ws.tmp_ns)
    mul!(ws.PTmp, transpose(T), N0)
    mul!(N0, ws.PTmp, T)
    mul!(ws.PTmp, transpose(T), N1)
    mul!(N1, ws.PTmp, T)
    mul!(ws.PTmp, transpose(T), N2)
    mul!(N2, ws.PTmp, T)
    =#
end
            
            
