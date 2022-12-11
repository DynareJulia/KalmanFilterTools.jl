using LinearAlgebra

function f(e)
    [ 1 1; 1 1+e]
end

function g(P)
    ones(1,2)*P*ones(2)
end

function u1(y, a0, P)
    v1 = y[1] - a0[1]
    @show v1
    F1 = P[1, 1]
    L = log(F1) + v1*v1/F1
    @show L
    K1 = 1
    a11 = a0[1] + v1
    P1 = P - P[:, 1]*P[:, 1]'./F1
    v2 = y[2] - a11
    F2 = P1[2, 2]
    @show v2, F2
    L += log(F2) + v2*v2/F2
end

function u2(y, a0, P)
    v = y - a0
    F = P
    L = log(det(F)) + v'*inv(F)*v
end

function u3(y, a0, P)
    v = y - a0
    F = P
    L = log(det(F)) + v'*pinv(F)*v
end

for i = 6:16
    e = 10.0^(-i)
    @show e
    try
        @show g(pinv(f(e)))
    catch
    end
    try
        @show g(inv(f(e)))
    catch
    end
    try
        @show ones(1,2)*ldiv!(cholesky(f(e)), ones(2))
    catch
    end
    y = [3, 1]
    a0 = [1, 1]
    @show "u1"
    try
        @show u1(y, a0, f(e))
    catch
    end
    @show "u2"
    try
        @show u2(y, a0, f(e))
    catch
    end
    @show "u3"
    try
        @show u3(y, a0, f(e))
    catch
    end
end
    
