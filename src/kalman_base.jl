function det_from_cholesky(achol::AbstractArray{T}) where {T<:AbstractFloat}
    x = 1.0
    @inbounds @simd for i = 1:size(achol, 1)
        x *= achol[i, i]
    end
    x * x
end

# alphah_t = a_t + P_t*r_{t-1}
function get_alphah!(alphah::AbstractVector{T}, a::AbstractVector{T}, P::AbstractArray{T}, r::AbstractVector{T}) where {T<:AbstractFloat}
    alphah .= a
    mul!(alphah, P, r, 1.0, 1.0)
end

# alphah_t = a_t + Pstar_t*r0_{t-1} + Pinf_t*r1_{t-1}     (DK 5.24)
function get_alphah!(valphah::AbstractVector{T}, va::AbstractVector{T},
    vPstar::AbstractArray{T}, vPinf::AbstractArray{T},
    r0::AbstractVector{T}, r1::AbstractVector{T}) where {T<:AbstractFloat}
    copy!(valphah, va)
    mul!(valphah, vPstar, r0, 1.0, 1.0)
    mul!(valphah, vPinf, r1, 1.0, 1.0)
end

function get_cholF!(cholF::AbstractArray{T}, F::AbstractArray{T}) where {T<:AbstractFloat}
    cholF .= 0.5 .* (F .+ transpose(F))
    cholF, info = LAPACK.potrf!('U', cholF)
    return info
end

# D = inv(F_t) + K_t*T*N_t*T'*K_t'
function get_D!(D, cholF, K, T, N, KT, tmp)
    copy!(D, cholF)
    LAPACK.potri!('U', D)
    # complete lower triangle
    n = size(D, 1)
    @inbounds for i = 1:n
        @simd for j = i:n
            D[j, i] = D[i, j]
        end
    end
    mul!(KT, K, T)
    mul!(tmp, KT, N)
    mul!(D, tmp, KT', 1.0, 1.0)
end


# D = inv(F_t) + K(DK)_t'*N_t*K(DK)_t (DK 4.69)
function get_D!(D::AbstractArray{U}, iF::AbstractArray{U}, K::AbstractArray{U}, N::AbstractArray{U}, tmp::AbstractArray{U}) where {U<:AbstractFloat}
    copy!(D, iF)
    mul!(tmp, transpose(K), N)
    mul!(D, tmp, K, 1.0, 1.0)
end

# D_t = KDK0_t'*N0_t*KDK0_t    (DK p. 135)
function get_D!(D::AbstractArray{T}, KDK0::AbstractArray{T}, N0::AbstractArray{T}, Tmp::AbstractArray{T}) where {T<:AbstractFloat}
    mul!(Tmp, Transpose(KDK0), N0)
    mul!(D, Tmp, KDK0)
end

# epsilon_h = H*(iF_t*v_t - K(DK)_t'*r_t) (DK 4.69)
function get_epsilonh!(epsilon::AbstractVector{U}, H::AbstractArray{U},
    iFv::AbstractVector{U}, K::AbstractArray{U},
    r::AbstractVector{U},
    tmp1::AbstractVector{U},
    tmp2::AbstractVector{U}) where {U<:AbstractFloat}
    copy!(tmp1, iFv)
    mul!(tmp1, transpose(K), r, -1.0, 1.0)
    mul!(epsilon, H, tmp1)
end

# epsilonh_t = H*(iF_t*v_t - K_t*T*r_t) Different K !!!
function get_epsilonh!(epsilon::AbstractVector{U}, H::AbstractArray{U},
    iFv::AbstractVector{U}, K::AbstractArray{U},
    T::AbstractArray{U}, r::AbstractVector{U},
    tmp1::AbstractVector{U},
    tmp2::AbstractVector{U}) where {U<:AbstractFloat}
    copy!(tmp1, iFv)
    mul!(tmp2, T, r)
    mul!(tmp1, K, tmp2, -1.0, 1.0)
    mul!(epsilon, H, tmp1)
end

# epsilon_t = -H_t*KDK0'*r0_t         (DK p. 135)
function get_epsilonh!(epsilon::AbstractVector{T}, H::AbstractArray{T},
    KDK0::AbstractArray{T}, r0::AbstractVector{T},
    tmp::AbstractVector{T}) where {T<:AbstractFloat}
    mul!(tmp, transpose(KDK0), r0)
    mul!(epsilon, H, tmp, -1.0, 0.0)
end

# etah = Q*R'*r_t
function get_etah!(etah::AbstractVector{T}, Q::AbstractArray{T},
    R::AbstractArray{T}, r::AbstractVector{T},
    tmp::AbstractVector{T}) where {T<:AbstractFloat}
    mul!(tmp, transpose(R), r)
    mul!(etah, Q, tmp)
end

function get_F(Zi, P, h, tmp)
    mul!(tmp, P, Zi)
    F = dot(Zi, tmp) + h
    return F
end

function get_F!(f::AbstractMatrix{T}, zp::AbstractArray{T}, z::AbstractArray{T}, p::AbstractMatrix{T}) where {T<:AbstractFloat}
    mul!(zp, z, p)
    mul!(f, zp, transpose(z))
end

function get_F!(f::AbstractMatrix{T}, zp::AbstractArray{T}, z::AbstractVector{U}, p::AbstractMatrix{T}) where {T<:AbstractFloat,U<:Integer}
    zp .= view(p, z, :)
    f .= view(zp, :, z)
end

# F = Z*P*Z' + H
function get_F!(f::AbstractMatrix{T}, zp::AbstractArray{T}, z::AbstractArray{T}, p::AbstractMatrix{T}, h::AbstractMatrix{T}) where {T<:AbstractFloat}
    copy!(f, h)
    mul!(zp, z, p)
    gemm!('N', 'T', 1.0, zp, z, 1.0, f)
end

# F = P(z,z) + H
function get_F!(f::AbstractMatrix{T}, zp::AbstractArray{T}, z::AbstractVector{I}, p::AbstractMatrix{T}, h::AbstractMatrix{T}) where {I<:Integer,T<:AbstractFloat}
    copy!(f, h)
    zp .= view(p, z, :)
    f .+= view(zp, :, z)
end

function get_Finf!(z::AbstractVector{T}, p::AbstractArray{T}, ukinf::AbstractVector{T}) where {T<:AbstractFloat}
    mul!(ukinf, p, z)
    Finf = BLAS.dot(z, ukinf)                         # F_{\infty,t} in 5.7 in DK (2012), relies on H being diagonal
end

function get_Finf!(z::U, p::AbstractArray{T}, ukinf::AbstractVector{T}) where {T<:AbstractFloat,U<:Integer}
    ukinf .= view(p, :, z)
    Finf = BLAS.dot(z, ukinf)                         # F_{\infty,t} in 5.7 in DK (2012), relies on H being diagonal
end

function get_Fstar!(z::AbstractVector{T}, p::AbstractArray{T}, h::T, uk1::AbstractVector{T}) where {T<:AbstractFloat}
    mul!(uk1, p, z)
    Finf = BLAS.dot(z, uk1) + h                 # F_{*,t} in 5.7 in DK (2012), relies on H being diagonal
end

function get_Finf!(z::U, p::AbstractArray{T}, h::T, uk1::AbstractVector{T}) where {T<:AbstractFloat,U<:Integer}
    uk1 .= view(p, :, z)
    Finf = BLAS.dot(z, uk1) + h                 # F_{*,t} in 5.7 in DK (2012), relies on H being diagonal
end

function get_iF!(iF::AbstractArray{T}, cholF::AbstractArray{T}) where {T<:AbstractFloat}
    copy!(iF, cholF)
    LAPACK.potri!('U', iF)
    n = size(iF, 1)
    for i = 1:n-1
        for j = 2:n
            iF[j, i] = iF[i, j]
        end
    end
end

function get_iFv!(iFv::AbstractVector{T}, cholF::AbstractArray{T}, v::AbstractVector{T}) where {T<:AbstractFloat}
    iFv .= v
    LAPACK.potrs!('U', cholF, iFv)
end

function get_iFZ!(iFZ::AbstractArray{T}, cholF::AbstractArray{T}, Z::AbstractArray{T}) where {T<:AbstractFloat}
    copy!(iFZ, Z)
    LAPACK.potrs!('U', cholF, iFZ)
end

function get_iFZ!(iFZ::AbstractArray{T}, cholF::AbstractArray{T}, z::AbstractVector{U}) where {T<:AbstractFloat,U<:Integer}
    n = length(z)
    fill!(iFZ, 0.0)
    @inbounds @simd for i = 1:n
        iFZ[i, z[i]] = 1.0
    end
    LAPACK.potrs!('U', cholF, iFZ)
end

# K = iF*Z*P
function get_K!(K::AbstractArray{T}, ZP::AbstractArray{T}, cholF::AbstractArray{T}) where {T<:AbstractFloat}
    copy!(K, ZP)
    LAPACK.potrs!('U', cholF, K)
end

# K1  = iFinf*(Z*Pstar - Fstar*K0)                           %(5.12) DK(2012);
function get_K1!(K1::AbstractArray{T}, Z::AbstractArray{T}, Pstar::AbstractArray{T}, Fstar::AbstractArray{T}, K0::AbstractArray{T}, cholF::AbstractArray{T}) where {T<:AbstractFloat}
    mul!(K1, Z, Pstar)
    gemm!('N', 'N', -1.0, Fstar, K0, 1.0, K1)
    LAPACK.potrs!('U', cholF, K1)
end

# K1  = iFinf*(Pstar[z,:] - Finf*K0)                           %(5.12) DK(2012);
function get_K1!(K1::AbstractArray{T}, z::AbstractVector{U}, Pstar::AbstractArray{T}, Finf::AbstractArray{T}, K0::AbstractArray{T}, cholF::AbstractArray{T}) where {T<:AbstractFloat,U<:Integer}
    K1 .= view(Pstar, z, :)
    gemm!('N', 'N', -1.0, Finf, K0, 1.0, K1)
    LAPACK.potrs!('U', cholF, K1)
end

# L_t = T - K(DK)_t*Z (DK 4.29)
function get_L!(L::AbstractArray{U}, T::AbstractArray{U}, K::AbstractArray{U}, Z::AbstractArray{U}) where {U<:AbstractFloat}
    copy!(L, T)
    mul!(L, K, Z, -1.0, 1.0)
end

# L_t = T - K(DK)_t*z (DK 4.29)
function get_L!(L::AbstractArray{U}, T::AbstractArray{U}, K::AbstractArray{U}, z::AbstractArray{W}) where {U<:AbstractFloat,W<:Integer}
    copy!(L, T)
    for j = 1:length(z)
        zj = z[j]
        for i = 1:size(L, 1)
            L[i, zj] -= K[i, j]
        end
    end
end

# L = T(I - K'*Z)
function get_L_alternative!(L::AbstractArray{U}, T::AbstractArray{U}, K::AbstractArray{U}, Z::AbstractArray{U}, Tmp::AbstractArray{U}) where {U<:AbstractFloat}
    fill!(Tmp, 0.0)
    @inbounds @simd for i = 1:size(Tmp, 1)
        Tmp[i, i] = 1.0
    end
    mul!(Tmp, transpose(K), Z, -1.0, 1.0)
    mul!(L, T, Tmp)
end

# L = T(I - K'*z)
function get_L!(L::AbstractArray{U}, T::AbstractArray{U}, K::AbstractArray{U}, z::AbstractArray{W}, Tmp::AbstractArray{U}) where {U<:AbstractFloat,W<:Integer}
    m, n = size(K)
    fill!(Tmp, 0.0)
    @inbounds for j = 1:m
        zj = z[j]
        @simd for k = 1:n
            Tmp[k, zj] = -K[j, k]
        end
    end
    @inbounds @simd for i = 1:n
        Tmp[i, i] += 1.0
    end
    mul!(L, T, Tmp)
end

# L1_t = - KDK*Z (DK 5.12)
function get_L1!(L1::AbstractArray{T}, KDK::AbstractArray{T}, Z::AbstractArray{T}) where {T<:AbstractFloat}
    mul!(L1, KDK, Z, -1.0, 0.0)
end

function get_L1!(L1::AbstractArray{T}, KDK::AbstractArray{T}, z::AbstractVector{U}) where {T<:AbstractFloat,U<:Integer}
    vL1 = view(L1, :, z)
    vL1 .= -KDK
end

function get_M!(y::AbstractArray{T}, x::AbstractArray{T}, work::AbstractArray{T}) where {T<:AbstractFloat}
    copy!(work, x)
    LAPACK.potri!('U', work)
    n = size(x, 1)
    # complete lower trianle and change sign of entire matrix
    @inbounds for i = 1:n
        @simd for j = 1:i-1
            y[j, i] = -work[j, i]
        end
        @simd for j = i:n
            y[j, i] = -work[i, j]
        end
    end
end

function get_prediction_error(Y::AbstractArray{T}, Z::AbstractArray{T}, a::AbstractVector{T}, i::U, t::U) where {T<:AbstractFloat,U<:Integer}
    Zi = view(Z, i, :)
    prediction_error = Y[i, t] - BLAS.dot(Zi, a)          # nu_{t,i} in 6.13 in DK (2012)
end

function get_prediction_error(Y::AbstractArray{T}, z::AbstractVector{U}, a::AbstractVector{T}, i::U, t::U) where {T<:AbstractFloat,U<:Integer}
    prediction_error = Y[i, t] - a[z[i]]                  # nu_{t,i} in 6.13 in DK (2012)
end

function get_QQ!(c::AbstractArray{T}, a::AbstractArray{T}, b::AbstractArray{T}, work::Array{T}) where {T<:AbstractFloat}
    mul!(work, a, b)
    mul!(c, work, transpose(a))
end

# att = a + K'*v
function get_updated_a!(att::AbstractVector{T}, a::AbstractVector{T}, K::AbstractArray{T}, v::AbstractVector{T}) where {T<:AbstractFloat}
    copy!(att, a)
    mul!(att, transpose(K), v, 1.0, 1.0)
end

function get_updated_Finfnonnull!(att,
    Pinftt,
    Pstartt,
    ZPinf,
    ZPstar,
    cholF,
    Fstar,
    Zsmall,
    Pstar,
    H,
    K0,
    K1,
    a,
    v,
    Pinf,
    PTmp)
    # K0   = iFinf*Z*Pinf                                   %define K0'=T^{-1}*K_0 with M_{\infty}=Pinf*Z'
    get_K!(K0, ZPinf, cholF)
    # Fstar  = Z*Pstar*Z' + H;                                        %(5.7) DK(2012)
    get_F!(Fstar, ZPstar, Zsmall, Pstar, H)
    # Kstar  = iFinf*(Z*Pstar - Fstar*K0)                           %(5.12) DK(2012);
    # note that there is a typo in DK (2003) with "+ K0" instead of "- K0",
    # but it is correct in their appendix
    get_K1!(K1, Zsmall, Pstar, Fstar, K0, cholF)
    # att = a + K0'*v                                                (5.13) DK(2012)
    get_updated_a!(att, a, K0, v)
    # Pinf_tt = Pinf - K0'*Z*Pinf                                    (5.14) DK(2012)
    get_updated_Ptt!(Pinftt, Pinf, K0, ZPinf)
    # Pstartt = Pstar-Pstar*Z'*K0-Pinf*Z'*K1                         (5.14) DK(2012)
    get_updated_Pstartt!(Pstartt, Pstar, ZPstar, K0, ZPinf,
        K1, Pinftt, PTmp)
end

function get_updated_Finfnull1!(a, Pinf, Pstar, ZPstar, cholF, Fstar, Z, H, T, K, QQ, a1, v, Ptmp, kalman_tol)

    # Fstar  = Z*Pstar*Z' + H;                                        (5.7) DK(2012)
    get_F!(Fstar, ZPstar, Z, Pstar, H)
    info = get_cholF!(cholF, Fstar)
    if info > 0 || rcond(Fstar) < kalman_tol
        throw(ErrorException("vFstar is singular"))
    end
    # K = iFstar*Z*Pstar                                              (5.15) DK(2012)
    get_K!(K, ZPstar, cholF)
    # a = T(a + K'*v)
    update_a!(a, K, v, a1, T)
    # Pinf = T*Pinf*T'
    update_P!(Pinf, T, QQ, Pinf, Ptmp)
    # Pstar = T*(Pstar - K'*Z*Pstar)*T'+ QQ
    update_P!(Pstar, T, QQ, K, ZPstar, Ptmp)
end

function get_updated_Finfnull2!(att,
    Pinftt,
    Pstartt,
    ZPinf,
    ZPstar,
    cholF,
    Fstar,
    Zsmall,
    Pstar,
    H,
    K,
    a,
    v,
    Pinf,
    kalman_tol)
    # Fstar  = Z*Pstar*Z' + H;                                        (5.7) DK(2012)
    get_F!(Fstar, ZPstar, Zsmall, Pstar, H)
    info = get_cholF!(cholF, Fstar)
    if info > 0 || rcond(Fstar) < kalman_tol
        throw(ErrorException("vFstar is singular"))
    end
    # K = iFstar*Z*Pstar                                              (5.15) DK(2012)
    get_K!(K, ZPstar, cholF)
    # att = a + K'*v                                                
    get_updated_a!(att, a, K, v)
    # Pinftt = Pinf
    Pinftt .= Pinf
    # Pstartt = Pstar - K'*Z*Pstar
    get_updated_Ptt!(Pstartt, Pstar, K, ZPstar)
end

# Ptt = P - K'*Z*P
function get_updated_Ptt!(Ptt::AbstractArray{T}, P::AbstractArray{T}, K::AbstractArray{T}, ZP::AbstractArray{T}) where {T<:AbstractFloat}
    copy!(Ptt, P)
    mul!(Ptt, transpose(K), ZP, -1.0, 1.0)
end

# Pstartt = Pstar-Pstar*Z'*K0-Pinf*Z'*K1                           %(5.14) DK(2012)
function get_updated_Pstartt!(Pstartt::AbstractArray{T}, Pstar::AbstractArray{T}, ZPstar::AbstractArray{T},
    K0::AbstractArray{T}, ZPinf::AbstractArray{T},
    K1::AbstractArray{T}, vPinftt::AbstractArray{T}, PTmp::AbstractArray{T}) where {T<:AbstractFloat}
    copy!(Pstartt, Pstar)
    mul!(Pstartt, transpose(ZPstar), K0, -1.0, 1.0)
    mul!(Pstartt, transpose(ZPinf), K1, -1.0, 1.0)
end

# v = y - Z*a -- basic
function get_v!(v::AbstractVector{T}, y::AbstractArray{V}, z::AbstractVecOrMat{T}, a::AbstractVector{T}, iy::U, ny::U) where {V<:Union{AbstractFloat,Missing},T<:AbstractFloat,U<:Integer}
    copyto!(v, 1, y, iy, ny)
    gemv!('N', -1.0, z, a, 1.0, v)
end

# v = y - Z*a -- basic -- univariate
function get_v!(Y::AbstractVector{V}, Z::AbstractVecOrMat{T}, a::AbstractVector{T}, i::U) where {V<:Union{AbstractFloat,Missing},T<:AbstractFloat,U<:Integer}
    v = Y[i]
    @inbounds @simd for j = 1:length(a)
        v -= Z[i, j] * a[j]
    end
    return v
end

# v = y - a[z] -- Z selection matrix
function get_v!(v::AbstractVector{T}, y::AbstractArray{V}, z::AbstractVector{U}, a::AbstractVector{T}, iy::U, ny::U) where {V<:Union{AbstractFloat,Missing},T<:AbstractFloat,U<:Integer}
    copyto!(v, 1, y, iy, ny)
    az = view(a, z)
    v .= v .- az
end

# v = y - a[z] -- Z selection matrix -- univariate
function get_v!(y::AbstractVector{V}, z::AbstractVector{U}, a::AbstractVector{T}, i::U) where {V<:Union{AbstractFloat,Missing},T<:AbstractFloat,U<:Integer}
    return y[i] - a[z[i]]
end

# v = y - Z*a -- missing observations
function get_v!(v::AbstractVector{T}, y::AbstractMatrix{V}, z::AbstractArray{T}, a::AbstractVector{T}, t::U, pattern::Vector{U}) where {V<:Union{AbstractFloat,Missing},T<:AbstractFloat,U<:Integer}
    v .= view(y, pattern, t)
    gemv!('N', -1.0, z, a, 1.0, v)
end

# v = y - a[z] -- Z selection matrix and missing variables
function get_v!(v::AbstractVector{T}, y::AbstractMatrix{V}, z::AbstractVector{U}, a::AbstractVector{T}, t::U, pattern::Vector{Int64}) where {V<:Union{AbstractFloat,Missing},T<:AbstractFloat,U<:Integer}
    v .= view(y, pattern, t) .- view(a, z)
end

# v = y - c - Z*a -- basic
function get_v!(v::AbstractArray{T}, y::AbstractArray{V}, c::AbstractVector{T}, z::AbstractArray{T}, a::AbstractArray{T}, iy::U, ny::U) where {V<:Union{AbstractFloat,Missing},T<:AbstractFloat,U<:Integer}
    copyto!(v, 1, y, iy, ny)
    v .-= c
    gemm!('N', 'N', -1.0, z, a, 1.0, v)
end

# v = y - c - Z*a -- basic -- univariate
function get_v!(Y::AbstractVector{V}, c::AbstractVector{T}, Z::AbstractVecOrMat{T}, a::AbstractVector{T}, i::U) where {V<:Union{AbstractFloat,Missing},T<:AbstractFloat,U<:Integer}
    v = Y[i] - c[i]
    @inbounds @simd for j = 1:length(a)
        v -= Z[i, j] * a[j]
    end
    return v
end

# v = y - c - a[z] -- Z selection matrix
function get_v!(v::AbstractVector{T}, y::AbstractArray{V}, c::AbstractVector{T}, z::AbstractVector{U}, a::AbstractArray{T}, iy::U, ny::U) where {V<:Union{AbstractFloat,Missing},T<:AbstractFloat,U<:Integer}
    copyto!(v, 1, y, iy, ny)
    az = view(a, z)
    v .-= c .+ az
end

# v = y - c - a[z] -- Z selection matrix -- univariate
function get_v!(y::AbstractVector{V}, c::AbstractVector{T}, z::AbstractVector{U}, a::AbstractVector{T}, i::U) where {V<:Union{AbstractFloat,Missing},T<:AbstractFloat,U<:Integer}
    return y[i] - c[i] - a[z[i]]
end

# v = y - c - Z*a -- missing observations
function get_v!(v::AbstractVector{T}, y::AbstractMatrix{V}, c::AbstractArray{T}, z::AbstractArray{T}, a::AbstractArray{T}, t::U, pattern::Vector{U}) where {V<:Union{AbstractFloat,Missing},T<:AbstractFloat,U<:Integer}
    v .= view(y, pattern, t) .- view(c, pattern)
    gemm!('N', 'N', -1.0, z, a, 1.0, v)
end

# v = y - c - a[z] -- Z selection matrix and missing variables
function get_v!(v::AbstractVector{T}, y::AbstractMatrix{V}, c::AbstractArray{T}, z::AbstractVector{U}, a::AbstractArray{T}, t::U, pattern::Vector{Int64}) where {V<:Union{AbstractFloat,Missing},T<:AbstractFloat,U<:Integer}
    v .= view(y, pattern, t) .- view(c, pattern) .- view(a, z)
end

# V_t = P_t - P_t*N_{t-1}*P_t
function get_Valpha!(V::AbstractArray{T}, P::AbstractArray{T}, N::AbstractArray{T}, Ptmp::AbstractArray{T}) where {T<:AbstractFloat}
    copy!(V, P)
    mul!(Ptmp, P, N)
    gemm!('N', 'N', -1.0, Ptmp, P, 1.0, V)
end

# Valpha_t = Pstar_t - Pstar_t*N0_{t-1}*Pstar_t
#            -(Pinf_t*N1_{t-1}*Pstar_t)' -Pinf_t*N1_{t-1}*Pstar_t -
#            -Pinf_t*N2_{t-1}*Pinf_t                      (DK 5.30)
function get_Valpha!(Valpha::AbstractArray{T},
    Pstar::AbstractArray{T},
    Pinf::AbstractArray{T}, N0::AbstractArray{T},
    N1::AbstractArray{T}, N2::AbstractArray{T},
    Tmp::AbstractArray{T}) where {T<:AbstractFloat}
    copy!(Valpha, Pstar)
    mul!(Tmp, Pstar, N0)
    mul!(Tmp, Pinf, N1, 1.0, 1.0)
    mul!(Valpha, Tmp, Pstar, -1.0, 1.0)
    mul!(Tmp, Pstar, N1)
    mul!(Tmp, Pinf, N2, 1.0, 1.0)
    mul!(Valpha, Tmp, Pinf, -1.0, 1.0)
end

# Vepsilon_t = H - H*D_t*H
function get_Vepsilon!(Vepsilon::AbstractArray{T}, H::AbstractArray{T}, D::AbstractArray{T}, tmp::AbstractArray{T}) where {T<:AbstractFloat}
    copy!(Vepsilon, H)
    mul!(tmp, H, D)
    mul!(Vepsilon, tmp, H, -1.0, 1.0)
end

# Veta_t = Q - Q*R'*N_t*R*Q
function get_Veta!(Veta, Q, R, N, RQ, tmp)
    copy!(Veta, Q)
    mul!(RQ, R, Q)
    mul!(tmp, N, RQ)
    mul!(Veta, transpose(RQ), tmp, -1.0, 1.0)
end

function get_vZsmall(Zsmall::AbstractArray{T}, iZsmall::AbstractVector{U}, Z::AbstractArray{T}, pattern::AbstractVector{U}, n::U, ny::U) where {T<:AbstractFloat,U<:Integer}
    vZsmall = view(Zsmall, 1:n, :)
    if n == ny
        copyto!(vZsmall, Z)
    else
        vZsmall .= view(Z, pattern, :)
    end
    return vZsmall
end

function get_vZsmall(Zsmall::AbstractArray{T}, iZsmall::AbstractVector{U}, z::AbstractArray{U}, pattern::AbstractVector{U}, n::U, ny::U) where {T<:AbstractFloat,U<:Integer}
    vZsmall = view(iZsmall, 1:n)
    if n == ny
        copyto!(vZsmall, z)
    else
        vZsmall .= view(z, pattern)
    end
    return vZsmall
end

function get_vZsmall(Zsmall::AbstractArray{T}, iZsmall::AbstractVector{U}, Z::AbstractArray{T}, pattern::AbstractVector{U}, n::U, ny::U, t::U) where {T<:AbstractFloat,U<:Integer}
    changeZ = ndims(Z) > 2
    vZ = changeZ ? view(Z, :, :, t) : view(Z, :, :)
    vZsmall = view(Zsmall, 1:n, :)
    if n == ny
        copyto!(vZsmall, vZ)
    else
        vZsmall .= view(vZ, pattern, :)
    end
    return vZsmall
end

function get_vZsmall(Zsmall::AbstractArray{T}, iZsmall::AbstractVector{U}, z::AbstractArray{U}, pattern::AbstractVector{U}, n::U, ny::U, t::U) where {T<:AbstractFloat,U<:Integer}
    changeZ = ndims(z) > 1
    vz = changeZ ? view(z, :, t) : view(z, :,)
    vZsmall = view(iZsmall, 1:n)
    if n == ny
        copyto!(vZsmall, vz)
    else
        vZsmall .= view(Z, pattern)
    end
    return vZsmall
end

"""
 rcond(cholF)

computes an approximate reverse condition number of a symmetric matrix, given its Cholesky decomposition:
``min l_{ii}/max l_{ii}`` (see N. Higham, 1987).
"""
function rcond(cholF::AbstractMatrix{T}) where {T<:Real}
    m1 = Inf
    m2 = 0.0
    n = size(cholF, 1)
    i = 1
    while i <= n * n
        x = cholF[i]
        x < m1 && (m1 = x)
        x > m2 && (m2 = x)
        i += n + 1
    end
    y = m1 / m2
    return y * y
end

# a = T(a + K'*v)
function update_a!(a::AbstractVector{U}, K::AbstractArray{U}, v::AbstractVector{U}, a1::Vector{U}, T::AbstractArray{U}) where {U<:AbstractFloat}
    copy!(a1, a)
    gemv!('T', 1.0, K, v, 1.0, a1)
    gemv!('N', 1.0, T, a1, 0.0, a)
end

# a = d + T(a + K'*v)
function update_a!(a1::AbstractArray{U}, a::AbstractArray, d::AbstractArray{U}, K::AbstractArray{U}, v::AbstractArray{U}, work::AbstractArray{U}, T::AbstractArray{U}) where {U<:AbstractFloat}
    copy!(work, a)
    gemm!('T', 'N', 1.0, K, v, 1.0, work)
    gemm!('N', 'N', 1.0, T, work, 0.0, a1)
    a1 .+= d
end

# a = d + T*att
function update_a!(a1::AbstractVector{U}, d::AbstractVector{U}, T::AbstractArray{U}, att::AbstractVector{U}) where {U<:AbstractFloat}
    copy!(a1, d)
    mul!(a1, T, att, 1.0, 1.0)
end

# a = d + a + K'*v
function filtered_a!(a1::AbstractArray{U}, a::AbstractArray{U}, d::AbstractArray{U}, K::AbstractArray{U}, v::AbstractArray{U}, work::AbstractArray{U}) where {U<:AbstractFloat}
    a1 .= a
    gemm!('T', 'N', 1.0, K, v, 1.0, a1)
    a1 .+= d
end

# K = K + Z*W*M*W'
function update_K!(K::AbstractArray{U}, ZWM::AbstractArray{U}, W::AbstractArray{U}) where {U<:AbstractFloat}
    gemm!('N', 'T', 1.0, ZWM, W, 1.0, K)
end

# M = M + M*W'*Z'iF*Z*W*M
function update_M!(M::AbstractArray{U}, Z::AbstractArray{U}, W::AbstractArray{U}, cholF::AbstractArray{U}, ZW::AbstractArray{U}, ZWM::AbstractArray{U}, iFZWM::AbstractArray{U}) where {U<:AbstractFloat}
    mul!(ZW, Z, W)
    mul!(ZWM, ZW, M)
    copy!(iFZWM, ZWM)
    LAPACK.potrs!('U', cholF, iFZWM)
    gemm!('T', 'N', 1.0, ZWM, iFZWM, 1.0, M)
end

# M = M + M*W(z,:)'*iF*W(z,:)*M
function update_M!(M::AbstractArray{U}, z::Vector{R}, W::AbstractArray{U}, cholF::AbstractArray{U}, ZW::AbstractArray{U}, ZWM::AbstractArray{U}, iFZWM::AbstractArray{U}) where {U<:AbstractFloat,R<:Real}
    ZW .= view(W, z, :)
    mul!(ZWM, ZW, M)
    copy!(iFZWM, ZWM)
    LAPACK.potrs!('U', cholF, iFZWM)
    gemm!('T', 'N', 1.0, ZWM, iFZWM, 1.0, M)
end

# N_{t-1} = Z_t'iF_t*Z_t + L_t'N_t*L_t
function update_N!(N1::AbstractArray{T}, Z::AbstractArray{T}, iFZ::AbstractArray{T}, L::AbstractArray{T}, N::AbstractArray{T}, Ptmp::AbstractArray{T}) where {T<:AbstractFloat}
    mul!(N1, transpose(Z), iFZ)
    mul!(Ptmp, transpose(L), N)
    mul!(N1, Ptmp, L, 1.0, 1.0)
end

function update_N!(N1::AbstractArray{T}, z::AbstractVector{U}, iFZ::AbstractArray{T}, L::AbstractArray{T}, N::AbstractArray{T}, Tmp::AbstractArray{T}) where {T<:AbstractFloat,U<:Integer}
    fill!(N1, 0.0)
    vN1 = view(N1, z, :)
    vN1 .= iFZ
    mul!(Tmp, transpose(L), N)
    mul!(N1, Tmp, L, 1.0, 1.0)
end

# N0_{t-1} = L0_t'N0_t*L0_t (DK 5.29)
function update_N0!(N0, L0, N0_1, PTmp)
    mul!(PTmp, transpose(L0), N0_1)
    mul!(N0, PTmp, L0)
end

# Finf = inv(Finf)
# N1_{t-1} = Z'*Finf*Z + L0'*N1_t*L0 + L1'*N0_t*L0
function update_N1!(N1::AbstractArray{T}, Z::AbstractArray{T},
    iFZ::AbstractArray{T}, L0::AbstractArray{T},
    N1_1::AbstractArray{T}, L1::AbstractArray{T},
    N0_1::AbstractArray{T}, PTmp::AbstractArray{T}) where {T<:AbstractFloat}
    mul!(N1, transpose(Z), iFZ)
    mul!(PTmp, transpose(L0), N1_1)
    mul!(PTmp, transpose(L1), N0_1, 1.0, 1.0)
    mul!(N1, PTmp, L0, 1.0, 1.0)
end

function update_N1!(N1::AbstractArray{T}, z::AbstractVector{U},
    iFZ::AbstractArray{T}, L0::AbstractArray{T},
    N1_1::AbstractArray{T}, L1::AbstractArray{T},
    N0_1::AbstractArray{T}, PTmp::AbstractArray{T}) where {T<:AbstractFloat,U<:Integer}
    vN1 = view(N1, z, :)
    vN1 .= iFZ
    mul!(PTmp, transpose(L0), N1_1)
    mul!(PTmp, transpose(L1), N0_1, 1.0, 1.0)
    mul!(N1, PTmp, L0, 1.0, 1.0)
end

# Fstar = -inv(Finf)*Fstar*inv(Finv)
# N2_{t-1} = Z'*Fstar*Z + L0'*N2_t*L0 + L0'*N1_t*L1
#            + L1'*N1_t*L0 + L1'*N0_t*L1
function update_N2!(N2::AbstractArray{T}, iFZ::AbstractArray{T},
    Fstar::AbstractArray{T}, L0::AbstractArray{T},
    N2_1::AbstractArray{T}, N1_1::AbstractArray{T},
    L1::AbstractArray{T}, N0_1::AbstractArray{T},
    Tmp1::AbstractArray{T}, Tmp2::AbstractArray{T}) where {T<:AbstractFloat}
    mul!(Tmp1, Fstar, iFZ)
    mul!(N2, transpose(iFZ), Tmp1, -1.0, 0.0)
    mul!(Tmp2, transpose(L0), N2_1)
    mul!(Tmp2, transpose(L1), N1_1, 1.0, 1.0)
    mul!(N2, Tmp2, L0, 1.0, 1.0)
    mul!(Tmp2, transpose(L0), N1_1)
    mul!(Tmp2, transpose(L1), N0_1, 1.0, 1.0)
    mul!(N2, Tmp2, L1, 1.0, 1.0)
end

# P = T*(P - K'*Z*P)*T'+ QQ
function update_P!(P::AbstractArray{U}, T::AbstractArray{U}, QQ::AbstractArray{U}, K::AbstractArray{U}, ZP::AbstractArray{U}, Ptmp::AbstractArray{U}) where {U<:AbstractFloat}
    gemm!('T', 'N', -1.0, K, ZP, 1.0, P)
    mul!(Ptmp, T, P)
    copy!(P, QQ)
    gemm!('N', 'T', 1.0, Ptmp, T, 1.0, P)
end

# P = P - K'*Z*P
function filtered_P!(P1::AbstractArray{U}, P::AbstractArray{U}, K::AbstractArray{U}, ZP::AbstractArray{U}, Ptmp::AbstractArray{U}) where {U<:AbstractFloat}
    copy!(P1, P)
    gemm!('T', 'N', -1.0, K, ZP, 1.0, P1)
end

# P = T*Ptt*T' + QQ
function update_P!(P::AbstractArray{U}, T::AbstractArray{U}, Ptt::AbstractArray{U}, QQ::AbstractArray{U}, Ptmp::AbstractArray{U}) where {U<:AbstractFloat}
    mul!(Ptmp, Ptt, transpose(T))
    copy!(P, QQ)
    mul!(P, T, Ptmp, 1.0, 1.0)
end

# Pinf = T*Pinftt*T'
function update_P!(P::AbstractArray{U}, T::AbstractArray{U}, Ptt::AbstractArray{U}, Ptmp::AbstractArray{U}) where {U<:AbstractFloat}
    mul!(Ptmp, Ptt, transpose(T))
    mul!(P, T, Ptmp)
end

# Pstar  = T*(Pstar-Pstar*Z'*K0-Pinf*Z'*K1)*T'+QQ         (5.14) DK(2012)
function update_Pstar!(Pstar, T, ZPinf, ZPstar, K0, K1, QQ, PTmp)
    copy!(PTmp, Pstar)
    mul!(PTmp, transpose(ZPstar), K0, -1.0, 1.0)
    mul!(PTmp, transpose(ZPinf), K1, -1.0, 1.0)
    copy!(Pstar, PTmp)
    mul!(PTmp, T, Pstar)
    copy!(Pstar, QQ)
    mul!(Pstar, PTmp, transpose(T), 1.0, 1.0)
end

# Pinf   = T*(Pinf-Pinf*Z'*K0)*T'                             (5.14) DK(2012)
function update_Pinf!(Pinf, T, ZPinf, K0, PTmp)
    mul!(Pinf, transpose(ZPinf), K0, -1.0, 1.0)
    mul!(PTmp, T, Pinf)
    mul!(Pinf, PTmp, transpose(T))
end

# r_{t-1} = Z_t'*iF_t*v_t + L_t'r_t
function update_r!(r1::AbstractVector{T}, Z::AbstractArray{T}, iFv::AbstractVector{T}, L::AbstractArray{T}, r::AbstractVector{T}) where {T<:AbstractFloat}
    mul!(r1, transpose(Z), iFv)
    gemm!('T', 'N', 1.0, L, r, 1.0, r1)
end

# r_{t-1} = z_t'*iF_t*v_t + L_t'r_t
function update_r!(r1::AbstractVector{T}, z::AbstractVector{U}, iFv::AbstractVector{T}, L::AbstractArray{T}, r::AbstractVector{T}) where {T<:AbstractFloat,U<:Integer}
    fill!(r1, 0.0)
    vr1 = view(r1, z)
    vr1 .= iFv
    gemm!('T', 'N', 1.0, L, r, 1.0, r1)
end

# r1_{t-1} = Z_t'*iFinf_t*v_t + L0_t'r1_t + L1_t'*r0_t      (DK 5.21)
function update_r!(r1::AbstractVector{T}, Z::AbstractArray{T},
    iFv::AbstractVector{T}, L0::AbstractArray{T},
    r1_1::AbstractVector{T}, L1::AbstractArray{T},
    r0_1::AbstractVector{T}) where {T<:AbstractFloat}
    mul!(r1, transpose(Z), iFv)
    mul!(r1, transpose(L0), r1_1, 1.0, 1.0)
    mul!(r1, transpose(L1), r0_1, 1.0, 1.0)
end

# r1_{t-1} = z_t'*iFinf_t*v_t + L0_t'r1_t + L1_t'*r0_t      (DK 5.21)
function update_r!(r1::AbstractVector{T}, z::AbstractVector{U},
    iFv::AbstractVector{T}, L0::AbstractArray{T},
    r1_1::AbstractVector{T}, L1::AbstractArray{T},
    r0_1::AbstractVector{T}) where {T<:AbstractFloat,U<:Integer}
    vr1 = view(r1, z)
    vr1 .= iFv
    mul!(r1, transpose(L0), r1_1, 1.0, 1.0)
    mul!(r1, transpose(L1), r0_1, 1.0, 1.0)
end

# W = T(W - K'*iF*Z*W)
function update_W!(W::AbstractArray{U}, ZW::AbstractArray{U}, cholF::AbstractArray{U}, T::AbstractArray{U}, K::AbstractArray{U}, iFZW::AbstractArray{U}, KtiFZW::AbstractArray{U}) where {U<:AbstractFloat}
    copy!(iFZW, ZW)
    LAPACK.potrs!('U', cholF, iFZW)
    copy!(KtiFZW, W)
    gemm!('T', 'N', -1.0, K, iFZW, 1.0, KtiFZW)
    mul!(W, T, KtiFZW)
end

