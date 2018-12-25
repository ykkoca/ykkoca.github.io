function kfilter(Y::Array, X0, F, R, Q, H, P0, s)

#=
% Input descriptions:

Signal Equation: Y = X*H + v
State Equation: X = X(-1)*F + w

R = v*v', mxm matrix
Q = w*w', kxk matrix

Y: nxm matrix,

X0: 1xk matrix
F: mxk matrix

H: kxm matrix

P0: kxk matrix
=#

n = size(Y,1)

Xu = Array{Array{Float64,2}}(n)
Xf = Array{Array{Float64,2}}(n)
Xs = Array{Array{Float64,2}}(n)

Pu = Array{Array{Float64,2}}(n)
Pf = Array{Array{Float64,2}}(n)


Xu[1] = X0
Pu[1] = P0

K = Pu[1]*H/(H'*Pu[1]*H + R)

Xf[1] = (Xu[1]' + K*(Y[1,:] - Xu[1]*H)')'
Pf[1] = Pu[1] - K*H'*Pu[1]

i = 1

# Forward Pass
while i < n

    Xu[i+1] = Xf[i]*F
    Pu[i+1] = F'*Pf[i]*F + Q

    K = Pu[i]*H*inv(H'*Pu[i]*H + R)

    Xf[i+1] = (Xu[i+1]' + K*(Y[i+1,:] - Xu[i+1]*H)')'
    Pf[i+1] = Pu[1] - K*H'*Pu[i]

    i += 1
end

#Smoother
if s == "smoother"

Ps = Array{Array{Float64,2}}(n)

Xs[n] = Xf[n]
Ps[n] = Pf[n]

for i = n-1:-1:1

    L = (Pf[i]*F)*inv(Pu[i+1])

    Xs[i] = (Xf[i]' + L*(Xs[i+1] - Xu[i+1])')'
    Ps[i] = Pf[i] + L*(Ps[i+1]-Pu[i+1])*L'

end

end

return Xf, Pf, Xs, Ps

end
