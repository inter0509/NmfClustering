%######################################################%
%##                                                  ##%
%##                                                  ##%
%##                                                  ##%
%######################################################%
function label = kmeans(fea, k)
% K-means clustering
%   fea: d x n data matrix
%   k: number of seeds
% Written by Ziyi Guo (zig312@lehigh.edu).

n = size(fea,2);
last = zeros(1,n);
label = ceil(k*rand(1,n));

while any(label ~= last)
    [u,~,label] = unique(label);   
    k = length(u);
	
	% convert clustering labels into a indicator matrix
    E = sparse(1:n,label,1,n,k,n);  
    m = fea*(E*spdiags(1./sum(E,1)',0,k,k));    
    last = label;
	
	% update cluster labels
    [~,l] = max(bsxfun(@minus,m'*fea,dot(m,m,1)'/2),[],1); 
    label = l';
end
[~,~,label] = unique(label);