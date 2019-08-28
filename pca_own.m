function [v,lamda]=pca_own(X)
% X must be of the form dimension x #observation
Xmean=repmat(mean(X,2),1,size(X,2));
Xnor=X-Xmean;
cvr=Xnor*Xnor';
[v,lamda]=eig(cvr); 
end
