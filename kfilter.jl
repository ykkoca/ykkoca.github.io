function kfilter(Y, X0, F, B, u, R, Q, H, P0, s)

#=
Input descriptions:

Signal Equation: Y = H*X + u
State Equation: X = F*X(-1) + u*B + w

R = u*u'
Q = w*w'

Y: nxm matrix,

X0: kx1 matrix
F: mxk matrix

B: mxk matrix
u: kx1 matrix

R: m*m matrix
Q: k*k matrix

H: k*k matrix

P0: k*k matrix
=#

n = size(Y,1)

Xu = Array{Float64}(size(X0,1),n)
Xf = Array{Float64}(size(X0,1),n)
Xs = Array{Float64}(size(X0,1),n)

Pu = Array{Array{Float64,2}}(n)
Pf = Array{Array{Float64,2}}(n)


Xu[:,1] = X0
Pu[1] = P0

K = Pu[1]*H'*inv(H*Pu[1]*H' + R)

Xf[:,1] = Xu[:,1] + K*(Y[1,:] - H*Xu[:,1])
Pf[1] = Pu[1] - K*H*Pu[1]

i = 1

# Forward Pass
while i < n

    Xu[:,i+1] = F*Xf[:,i] #+ B*u[:,i]
    Pu[i+1] = F*Pf[i]*F' + Q

    K = Pu[i]*H'*inv(H*Pu[i]*H' + R)

    Xf[:,i+1] = Xu[:,i+1] + K*(Y[i+1,:] - H*Xu[:,i+1])
    Pf[i+1] = Pu[1] - K*H*Pu[i]

    i += 1
end

Ps = Array{Array{Float64,2}}(n)

Xs[:,n] = Xf[:,n]
Ps[n] = Pf[n]

if s == "smoother"

for i = n-1:-1:1

    L = Pf[i]*F'*inv(Pu[i+1])

    Xs[:,i] = Xf[:,i] + L*(Xs[:,i+1] - Xu[:,i+1])
    Ps[i] = Pf[i] + L*(Ps[i+1]-Pu[i+1])*L'


end

end

return Xf, Pf, Xs, Ps

end
