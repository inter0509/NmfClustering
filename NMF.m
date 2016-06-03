%######################################################%
%##                                                  ##%
%##                                                  ##%
%##                                                  ##%
%######################################################%
function [W,H] = NMF(V,r,maxiter)
% Perform non-negative matrix factorization, NMF.
%   V: m x n data matrix
%   r: the reduced dimension
%   maxiter: the maximum number of iterations
%   W: m x r basis matrix
%   H: r x n coefficient matrix
%
%  NMF algorithm paper: Lee, Daniel D., and H. Sebastian Seung. "Algorithms for non-negative matrix factorization." 
%                        Advances in neural information processing systems. 2001.
%  Written by Ziyi Guo (zig312@lehigh.edu).

if min(min(V)) < 0
    error('Matrix entries can not be negative');
end

[n,m] = size(V);

W = abs(rand(n,r));
W = W./(ones(n,1)*sum(W));
H = abs(rand(r,m));
eps = 1e-9;

% NMF iteratively updates
for iter=1:maxiter
  H = H.*((W'*V)./((W'*W)*H+eps));
  W = W./(ones(n,1)*sum(W));
  W = W.*((V*H')./(W*(H*H')+eps));
end