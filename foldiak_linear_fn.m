function [T,Thist,Q,W, InfoTransferRatio] = foldiak_linear_fn(X, alphaa, betaa, output_neuron_nmbr, maxiter)

X=removemean(X);
[N K] = size(X); %N --> Dimension    K---> # of samples

%% initialize weights
Q=rand(N,output_neuron_nmbr)-0.5;   % Feedforward weights 
W=zeros(output_neuron_nmbr,output_neuron_nmbr); %Lateral weights
Y=zeros(output_neuron_nmbr,K);
%% Initiaze from previous weights for retraining
% Q=Qold;   % Feedforward weights 
% W=Wold; %Lateral weights
% Y=zeros(output_neuron_nmbr,K);

%% Computing mutual information transfer 
[v,l] = pca_own(X);
P = v(:,end:-1:end-(output_neuron_nmbr-1)); P = P';
CX = X*X';
Cp = P * CX *P';
Ipca = 0.5 * log(det(Cp));
Icurr = 0; tol = 0.84;

%% Main loop
% figure
ii = 1;
convergeflag = 1;
count=1;
while Icurr<= tol*Ipca
     T = inv(eye(output_neuron_nmbr)-W)*Q';
     Thist{count}=T;
    CY = T * CX * T';
%     CY = Y*Y';
    offdiagCY = CY;  offdiagCY(logical(eye(size(offdiagCY)))) = 0;
    diagCY = CY.*eye(output_neuron_nmbr);
    W = W - alphaa*offdiagCY;
    Q = Q + (betaa*(T*CX - diagCY*Q'))'; 
    I(ii) = 0.5 *log(det(CY)/det(T*T'));
    InfoTransferRatio(ii) = abs(I(ii))/Ipca;
    InfoTransferRatio(ii)
%     plot(abs(I)/Ipca); title('I/Ipca')
%     pause(0.01)
    
    Icurr = I(ii);
    ii = ii + 1;
    if ii > maxiter
        Icurr = Ipca;
        convergeflag = 0;
    end
    count=count+1;
end
if convergeflag == 1
    fprintf('Foldiac network weights converged in %d iterations',ii);
else fprintf('Foldiac network weights did not converge in %d iterations',maxiter);
end