function [Xf, Pf, Xs, Ps] = kfilter(Y, X0, F, R, Q, H, P0, s)

% Input descriptions:

% Signal Equation: Y = X*H + v
% State Equation: X = X(-1)*F + w

% R = v*v', mxm matrix
% Q = w*w', kxk matrix

% Y: nxm matrix,

% X0: 1xk matrix
% F: mxk matrix

% H: kxm matrix

% P0: kxk matrix

Xs = [];
Ps = [];

n = size(Y,1);

Xu(1,:) = X0;
Pu(1,:,:) = P0;

K = squeeze(Pu(1,:,:))*H/(H'*squeeze(Pu(1,:,:))*H + R);

Xf(1,:) = (Xu(1,:)' + K*(Y(1,:) - Xu(1,:)*H)')';
Pf(1,:,:) = squeeze(Pu(1,:,:)) - K*H'*squeeze(Pu(1,:,:));

i=1;

% Forward Pass
while i < n

    Xu(i+1,:) = Xf(i,:)*F; % + B*u(:,i);
    Pu(i+1,:,:) = F'*squeeze(Pf(i,:,:))*F + Q;

    K = squeeze(Pu(i,:,:))*H/(H'*squeeze(Pu(i,:,:))*H + R);

    Xf(i+1,:) = Xu(i+1,:)' + K*(Y(i+1,:) - Xu(i+1,:)*H)';
    Pf(i+1,:,:) = squeeze(Pu(1,:,:)) - K*H'*squeeze(Pu(i,:,:));

    i = i+1;
end

if strcmpi(s, 'smoother')
% Smoother (Rauch-Tung-Strießel)
Xs(n,:) = Xf(n,:);
Ps(n,:,:) = squeeze(Pf(n,:,:));

for i = n-1:-1:1


    L = (squeeze(Pf(i,:,:))*F)/squeeze(Pu(i+1,:,:));

    Xs(i,:) = (Xf(i,:)' + L*(Xs(i+1,:) - Xu(i+1,:))')';
    Ps(i,:,:) = squeeze(Pf(i,:,:)) + L*(squeeze(Ps(i+1,:,:))-squeeze(Pu(i+1,:,:)))*L';

end

end

end
