function [Xf, Pf, Xs, Ps] = kfilter(Y, X0, F, B, u, R, Q, H, P0, s)

% Input descriptions:

% Signal Equation: Y = H*X + u
% State Equation: X = F*X(-1) + u*B + w

% R = u*u'
% Q = w*w'

% Y: nxm matrix, 

% X0: kx1 matrix
% F: mxk matrix

% B: mxk matrix
% u: kx1 matrix

% R: m*m matrix
% Q: k*k matrix

% H: k*k matrix

%P0: k*k matrix

Xs = [];
Ps = [];

n = size(Y,1);

Xu(:,1) = X0;
Pu(1,:,:) = P0;

K = squeeze(Pu(1,:,:))*H'*inv(H*squeeze(Pu(1,:,:))*H' + R);

Xf(:,1) = Xu(:,1) + K*(Y(1,:) - H*Xu(:,1));
Pf(1,:,:) = squeeze(Pu(1,:,:)) - K*H*squeeze(Pu(1,:,:));

i=1;

% Forward Pass
while i < n
    
    Xu(:,i+1) = F*Xf(:,i) + B*u(:,i);
    Pu(i+1,:,:) = F*squeeze(Pf(i,:,:))*F' + Q;
    
    K = squeeze(Pu(i,:,:))*H'*inv(H*squeeze(Pu(i,:,:))*H' + R);
    
    Xf(:,i+1) = Xu(:,i+1) + K*(Y(i+1,:) - H*Xu(:,i+1));
    Pf(i+1,:,:) = squeeze(Pu(1,:,:)) - K*H*squeeze(Pu(i,:,:));
    
    i = i+1;
end

if strcmpi(s, 'smoother')

Xs(:,n) = Xf(:,n);
Ps(n,:,:) = squeeze(Pf(n,:,:));
    
for i = n-1:-1:1     

    
    L = squeeze(Pf(i,:,:))*F'*inv(H*squeeze(Pf(i,:,:))*H' + R);
    
    Xs(:,i) = Xf(:,i) + L*(Xs(:,i+1) - Xu(:,i+1));
    Ps(i,:,:) = squeeze(Pf(i,:,:)) + L*(squeeze(Ps(i+1,:,:))-squeeze(Pu(i+1,:,:)))*L';
    
    i = i+1;
    
end

end
    
end