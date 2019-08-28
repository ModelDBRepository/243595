function [Y] = removemean(X)
% X must be of the form dimension x #observation
Xmean=repmat(mean(X,2),1,size(X,2));
Y=X-Xmean;
end
