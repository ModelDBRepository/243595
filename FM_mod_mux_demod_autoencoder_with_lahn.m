% This code implements an oscillatory autoencoder model that accepts a 4
% dimensional simulated input and compressess to 2 dimensional signal and
% tries to reconstrcut back the original signal.
clc
clear all
close all
%% Signal generation
Fs=6000; Ts=1/Fs; 
t0 = 15;                           % signal duration
ts = Ts;                            % sampling interval
fc1 = 200;                        	% carrier frequency
fc2 = 350;
fc3 = 850;
fc4 = 1000;
clc

kf = 50;                         	% modulation index
fs = 1/ts;                       	% sampling frequency
t = [0:ts:t0-ts];                   	% time vector
df = 0.25;                          % required frequency resolution
carrier_sig1 = sin(2*pi*fc1*t);       % carrier signal
carrier_sig2 = sin(2*pi*fc2*t);
carrier_sig3 = sin(2*pi*fc3*t);
carrier_sig4 = sin(2*pi*fc4*t);
% msg = [1*ones(1,t0/(3*ts)),-2*ones(1,t0/(3*ts)),zeros(1,t0/(3*ts)+1)];
msg1=sin(2*pi*5*t)+0.5*sin(2*pi*6*t);
msg2=sin(2*pi*10*t)+0.5*sin(2*pi*14*t);
msg3=sin(2*pi*25*t)+0.5*sin(2*pi*28*t);
msg4=sin(2*pi*35*t)+0.5*sin(2*pi*40*t);
% figure;subplot(2,1,1); plot(msg1); title('Message signal1')
% subplot(2,1,2); plot(msg2); title('Message signal2')
%% FM Modulation
int_msg(1) = 0;
fm_phase_dot1=0; fm_phase_dot2=0; fm_phase_dot3=0; fm_phase_dot4=0;
fm_phase1 = fm_phase_dot1*Ts; fm_phase2 = fm_phase_dot2*Ts; fm_phase3 = fm_phase_dot3*Ts; fm_phase4 = fm_phase_dot4*Ts;
for ii = 2 : length(t)-1                  	
    fm_phase_dot1(ii) = 2*pi*fc1 + 2*pi*kf*msg1(ii); 
    fm_phase_dot2(ii) = 2*pi*fc2 + 2*pi*kf*msg2(ii);
    fm_phase_dot3(ii) = 2*pi*fc3 + 2*pi*kf*msg3(ii);
    fm_phase_dot4(ii) = 2*pi*fc4 + 2*pi*kf*msg4(ii);
    fm_phase1(ii)=fm_phase1(ii-1) + fm_phase_dot1(ii)*Ts;
    fm_phase2(ii)=fm_phase2(ii-1) + fm_phase_dot2(ii)*Ts;
    fm_phase3(ii)=fm_phase3(ii-1) + fm_phase_dot3(ii)*Ts;
    fm_phase4(ii)=fm_phase4(ii-1) + fm_phase_dot4(ii)*Ts;
end
fm_sig1=sin(fm_phase1);   % modulated signal
fm_sig2=sin(fm_phase2);   % modulated signal
fm_sig3=sin(fm_phase3);   % modulated signal
fm_sig4=sin(fm_phase4);   % modulated signal
fm_sig1(end+1)=fm_sig1(end); fm_sig2(end+1)=fm_sig2(end); fm_sig3(end+1)=fm_sig3(end); fm_sig4(end+1)=fm_sig4(end);

tstart=6000;
figure; subplot(2,2,1);plot(t(end-tstart:end),msg1(end-tstart:end));xlim([t(end-tstart) t(end)]); title('message signal1')
subplot(2,2,2);plot(t(end-tstart:end),msg2(end-tstart:end));xlim([t(end-tstart) t(end)]); title('message signal2')
subplot(2,2,3);plot(t(end-tstart:end),msg3(end-tstart:end));xlim([t(end-tstart) t(end)]); title('message signal3')
subplot(2,2,4);plot(t(end-tstart:end),msg4(end-tstart:end));xlim([t(end-tstart) t(end)]); title('message signal4')
%% MUX usking k nodes
X=[fm_sig1' fm_sig2' fm_sig3' fm_sig4'];
PI1d = X';
[N K] = size(PI1d); %N --> Dimension    K---> # of samples

%MUX using PCA
[v,lamda]=pca(X);
wt = v(:,1:2); %extract first two pcs

%MUX using network of PCA subspace learning (if using this comment above analytical PCA)
%load saved wts (comment the below line if lahn is getting trained using the foldiak_linear_fn)
% load('lahn_wts_saved_2')
% wt = T';

%train lahn (uncomment the below lines to train lahn)
% PI1d=removemean(PI1d);
% alphaa = 0.000001/K;
% betaa = 0.000001/K; 
% output_neuron_nmbr = 2;
% maxiter = 2000000;
% [T,InfoTransferRatio] = foldiak_linear_fn(PI1d, alphaa, betaa, output_neuron_nmbr, maxiter);

%wt(:,1)=wt(:,1)/norm(wt(:,1));wt(:,2)=wt(:,2)/norm(wt(:,2));
MUX(:,1) = wt(:,1)'*X'; MUX(:,2) = wt(:,2)'*X';  MUX1=MUX(:,1);MUX2=MUX(:,2);

%% FFT of the signal
L = length(MUX)*Ts;     % signal duration
t = 0:1/Fs:L-1/Fs; % Time vector
f = -(Fs-1/L)/2:1/L:(Fs-1/L)/2;  % Frequency vector

fft1 = abs((2/Fs)*fft(fm_sig1));
fft2 = abs((2/Fs)*fft(fm_sig2));
fft3 = abs((2/Fs)*fft(fm_sig3));
fft4 = abs((2/Fs)*fft(fm_sig4));
fftMUX1 = abs((2/Fs)*fft(MUX1));
fftMUX2 = abs((2/Fs)*fft(MUX2));

figure
subplot(6,1,1); plot(f,fftshift(fft1),'Linewidth',2);xlim([0 max(f)/2]);title('FFT of FM1')
subplot(6,1,2); plot(f,fftshift(fft2),'Linewidth',2);xlim([0 max(f)/2]);title('FFT of FM2')
subplot(6,1,3); plot(f,fftshift(fft3),'Linewidth',2);xlim([0 max(f)/2]);title('FFT of FM3')
subplot(6,1,4); plot(f,fftshift(fft4),'Linewidth',2);xlim([0 max(f)/2]);title('FFT of FM4')
subplot(6,1,5); plot(f,fftshift(fftMUX1),'Linewidth',2);xlim([0 max(f)/2]);title('FFT of Composite signal1');
subplot(6,1,6); plot(f,fftshift(fftMUX2),'Linewidth',2);xlim([0 max(f)/2]);title('FFT of Composite signal2');
xlabel('Frequency'); ylabel('Amplitude')

%% Demux using Adaptive Hopf dynamics
bf1 = 1*fc1-1*fc1/2;  bf2 = 1*fc2+1*fc2/2; bf3 = 1*fc3-1*fc3/8; bf4 = 1*fc4+1*fc4/4;
omega1 = 2*pi*bf1;  omega2 = 2*pi*bf2; omega3 = 2*pi*bf3;  omega4 = 2*pi*bf4;
r1=1;   r2=1;   r3=1;   r4=1;
phi1=rand;  phi2=rand;  phi3=rand;  phi4=rand;
epsl1 = 0.9;    epsl2 = 0.9;    epsl3 = 0.9;    epsl4 = 0.9;
dt=Ts;
A1=1000; A2=2000; A3=2000; A4=1500;    
% wt_from_MUX_to_Hopf = [1 0;0 1;0 1;1 0]';
% wt_from_MUX_to_Hopf = wt';
wt_from_MUX_to_Hopf = pinv(wt);
MUX_to_Hopf1 = wt_from_MUX_to_Hopf(:,1)'*MUX';MUX_to_Hopf1=MUX_to_Hopf1';
MUX_to_Hopf2 = wt_from_MUX_to_Hopf(:,2)'*MUX';MUX_to_Hopf2=MUX_to_Hopf2';
MUX_to_Hopf3 = wt_from_MUX_to_Hopf(:,3)'*MUX';MUX_to_Hopf3=MUX_to_Hopf3';
MUX_to_Hopf4 = wt_from_MUX_to_Hopf(:,4)'*MUX';MUX_to_Hopf4=MUX_to_Hopf4';
for ii=2:length(MUX)
    rdot1 = r1(ii-1)*(1-r1(ii-1)^2);
    phidot1 = omega1(ii-1) - A1*(epsl1/r1(ii-1))*MUX_to_Hopf1(ii-1)*sin(phi1(ii-1));
    omegadot1 = -A1*epsl1*MUX_to_Hopf1(ii-1)*sin(phi1(ii-1));
    r1(ii) = r1(ii-1)+rdot1*dt;
    phi1(ii) = phi1(ii-1) + phidot1*dt;
    omega1(ii) = omega1(ii-1)+omegadot1*dt;
    
    rdot2 = r2(ii-1)*(1-r2(ii-1)^2);     
    phidot2 = omega2(ii-1) - A2*(epsl2/r2(ii-1))*MUX_to_Hopf2(ii-1)*sin(phi2(ii-1));
    omegadot2 = -A2*epsl2*MUX_to_Hopf2(ii-1)*sin(phi2(ii-1));
    r2(ii) = r2(ii-1)+rdot2*dt;
    phi2(ii) = phi2(ii-1) + phidot2*dt;
    omega2(ii) = omega2(ii-1)+omegadot2*dt;
    
    rdot3 = r3(ii-1)*(1-r3(ii-1)^2);     
    phidot3 = omega3(ii-1) - A3*(epsl3/r3(ii-1))*MUX_to_Hopf3(ii-1)*sin(phi3(ii-1));
    omegadot3 = -A3*epsl3*MUX_to_Hopf3(ii-1)*sin(phi3(ii-1));
    r3(ii) = r3(ii-1)+rdot3*dt;
    phi3(ii) = phi3(ii-1) + phidot3*dt;
    omega3(ii) = omega3(ii-1)+omegadot3*dt;
    
    rdot4 = r4(ii-1)*(1-r4(ii-1)^2);     
    phidot4 = omega4(ii-1) - A4*(epsl4/r4(ii-1))*MUX_to_Hopf4(ii-1)*sin(phi4(ii-1));
    omegadot4 = -A4*epsl4*MUX_to_Hopf4(ii-1)*sin(phi4(ii-1));
    r4(ii) = r4(ii-1)+rdot4*dt;
    phi4(ii) = phi4(ii-1) + phidot4*dt;
    omega4(ii) = omega4(ii-1)+omegadot4*dt;
end
x1=r1.*cos(phi1);   x2=r2.*cos(phi2);   x3=r3.*cos(phi3);   x4=r4.*cos(phi4);
y1=r1.*sin(phi1);   y2=r2.*sin(phi2);   y3=r3.*sin(phi3);   y4=r4.*sin(phi4);
figure; plot(t,omega1/(2*pi),'Linewidth',2);  hold on; xlabel('Time'); ylabel('Frequency of Hopf oscillator')
plot(t,omega2/(2*pi),'r','Linewidth',2);  
plot(t,omega3/(2*pi),'g','Linewidth',2);  
plot(t,omega4/(2*pi),'k','Linewidth',2);  
legend('Frequency of HO1','Frequency of HO2','Frequency of HO3','Frequency of HO4')
%%
figure; subplot(2,2,1);plot(t(end-tstart:end),y1(end-tstart:end));xlim([t(end-tstart) t(end)]); title('Hopf oscillations1')
subplot(2,2,2); plot(t(end-tstart:end),y2(end-tstart:end));xlim([t(end-tstart) t(end)]); title('Hopf oscillations2')
subplot(2,2,3); plot(t(end-tstart:end),y3(end-tstart:end));xlim([t(end-tstart) t(end)]); title('Hopf oscillations3')
subplot(2,2,4); plot(t(end-tstart:end),y4(end-tstart:end));xlim([t(end-tstart) t(end)]); title('Hopf oscillations4')

fft_y1 = abs((2/Fs)*fft(y1));
fft_y2 = abs((2/Fs)*fft(y2));
fft_y3 = abs((2/Fs)*fft(y3));
fft_y4 = abs((2/Fs)*fft(y4));

figure; subplot(6,1,1);plot(f,fftshift(fftMUX1),'Linewidth',2);xlim([0 max(f)/2]);title('FFT of Composite signal1'); 
subplot(6,1,2);plot(f,fftshift(fftMUX2),'Linewidth',2);xlim([0 max(f)/2]);title('FFT of Composite signal2'); 
subplot(6,1,3); plot(f,fftshift(fft_y1));xlim([0 max(f)/2]); title('FFT of HO1') 
subplot(6,1,4);  plot(f,fftshift(fft_y2),'Linewidth',2);xlim([0 max(f)/2]); title('FFT of HO2') 
subplot(6,1,5);  plot(f,fftshift(fft_y3),'Linewidth',2);xlim([0 max(f)/2]); title('FFT of HO3') 
subplot(6,1,6);  plot(f,fftshift(fft_y4),'Linewidth',2);xlim([0 max(f)/2]); title('FFT of HO4') 
xlabel('Frequency'); ylabel('Amplitude')
%% Demodulation using PLL
VCO_out_gain = 1;
VCO_inp_gain = Fs;
LPF_Gain = 1;

LPF_out1 = 0.0*rand;    LPF_out2 = 0.0*rand;    LPF_out3 = 0.0*rand;    LPF_out4 = 0.0*rand;     
VCO_phase_dot1 = 0;     VCO_phase_dot2 = 0; VCO_phase_dot3 = 0;     VCO_phase_dot4 = 0; 
VCO_phase1 = VCO_phase_dot1*Ts; VCO_phase2 = VCO_phase_dot2*Ts; VCO_phase3 = VCO_phase_dot3*Ts; VCO_phase4 = VCO_phase_dot4*Ts;
VCO_out1 = VCO_out_gain*cos(VCO_phase1);    VCO_out2 = VCO_out_gain*cos(VCO_phase2);    VCO_out3 = VCO_out_gain*cos(VCO_phase3);    VCO_out4 = VCO_out_gain*cos(VCO_phase4);

for ii=2:length(t)-1
    VCO_phase_dot1(ii) = 2*pi*fc1 + VCO_inp_gain*LPF_out1(ii-1);
    VCO_phase1(ii) = VCO_phase1(ii-1) + VCO_phase_dot1(ii)*Ts;
    VCO_out1(ii) = VCO_out_gain*cos(VCO_phase1(ii));
    LPF_out1(ii) = LPF_Gain*sin(phi1(ii)-VCO_phase1(ii));
    
    VCO_phase_dot2(ii) = 2*pi*fc2 + VCO_inp_gain*LPF_out2(ii-1);
    VCO_phase2(ii) = VCO_phase2(ii-1) + VCO_phase_dot2(ii)*Ts;
    VCO_out2(ii) = VCO_out_gain*cos(VCO_phase2(ii));
    LPF_out2(ii) = LPF_Gain*sin(phi2(ii)-VCO_phase2(ii));
    
    VCO_phase_dot3(ii) = 2*pi*fc3 + VCO_inp_gain*LPF_out3(ii-1);
    VCO_phase3(ii) = VCO_phase3(ii-1) + VCO_phase_dot3(ii)*Ts;
    VCO_out3(ii) = VCO_out_gain*cos(VCO_phase3(ii));
    LPF_out3(ii) = LPF_Gain*sin(phi3(ii)-VCO_phase3(ii));
    
    VCO_phase_dot4(ii) = 2*pi*fc4 + VCO_inp_gain*LPF_out4(ii-1);
    VCO_phase4(ii) = VCO_phase4(ii-1) + VCO_phase_dot4(ii)*Ts;
    VCO_out4(ii) = VCO_out_gain*cos(VCO_phase4(ii));
    LPF_out4(ii) = LPF_Gain*sin(phi4(ii)-VCO_phase4(ii));
end
LPF_out1(end+1)=LPF_out1(end); LPF_out2(end+1)=LPF_out2(end);   LPF_out3(end+1)=LPF_out3(end); LPF_out4(end+1)=LPF_out4(end);
LPF_out1 = LPF_out1/max(LPF_out1);  LPF_out2 = LPF_out2/max(LPF_out2);  LPF_out3 = LPF_out3/max(LPF_out3);  LPF_out4 = LPF_out4/max(LPF_out4);

%% LPF (low pass filter) out to Leaky Integrator
niter=length(LPF_out1);
wc1=2*pi*6; wc2=2*pi*10;    wc3=2*pi*30; wc4=2*pi*40;
LIF1_HO1(1)=0; LIF2_HO1(1)=0;   
LIF1_HO2(1)=0; LIF2_HO2(1)=0;
LIF1_HO3(1)=0; LIF2_HO3(1)=0;
LIF1_HO4(1)=0; LIF2_HO4(1)=0;
dt=Ts;
for ii=2:niter
    dLIF1_HO1=wc1*(-LIF1_HO1(ii-1)+LPF_out1(ii));
    LIF1_HO1(ii)=LIF1_HO1(ii-1)+dLIF1_HO1*dt;
    dLIF1_HO2=wc2*(-LIF1_HO2(ii-1)+LPF_out2(ii));
    LIF1_HO2(ii)=LIF1_HO2(ii-1)+dLIF1_HO2*dt;
    dLIF1_HO3=wc3*(-LIF1_HO3(ii-1)+LPF_out3(ii));
    LIF1_HO3(ii)=LIF1_HO3(ii-1)+dLIF1_HO3*dt;
    dLIF1_HO4=wc4*(-LIF1_HO4(ii-1)+LPF_out4(ii));
    LIF1_HO4(ii)=LIF1_HO4(ii-1)+dLIF1_HO4*dt;
        
    dLIF2_HO1=wc1*(-LIF2_HO1(ii-1)+LIF1_HO1(ii));
    LIF2_HO1(ii)=LIF2_HO1(ii-1)+dLIF2_HO1*dt;    
    dLIF2_HO2=wc2*(-LIF2_HO2(ii-1)+LIF1_HO2(ii));
    LIF2_HO2(ii)=LIF2_HO2(ii-1)+dLIF2_HO2*dt;    
    dLIF2_HO3=wc3*(-LIF2_HO3(ii-1)+LIF1_HO3(ii));
    LIF2_HO3(ii)=LIF2_HO3(ii-1)+dLIF2_HO3*dt;    
    dLIF2_HO4=wc4*(-LIF2_HO4(ii-1)+LIF1_HO4(ii));
    LIF2_HO4(ii)=LIF2_HO4(ii-1)+dLIF2_HO4*dt;    
end

figure;subplot(2,1,1); plot(t(end-tstart:end),msg1(end-tstart:end),'Linewidth',2);xlim([t(end-tstart) t(end)]);title('Message signal1');
subplot(2,1,2); plot(t(end-tstart:end),LIF2_HO1(end-tstart:end),'Linewidth',2);xlim([t(end-tstart) t(end)]);ylim([min(LIF2_HO1(end-tstart:end))-0.03 max(LIF2_HO1(end-tstart:end))+0.03]);title('Demodulated signal1') 
figure;subplot(2,1,1); plot(t(end-tstart:end),msg2(end-tstart:end),'Linewidth',2);xlim([t(end-tstart) t(end)]);title('Message signal2');
subplot(2,1,2); plot(t(end-tstart:end),LIF2_HO2(end-tstart:end),'Linewidth',2);xlim([t(end-tstart) t(end)]);ylim([min(LIF2_HO2(end-tstart:end))-0.02 max(LIF2_HO2(end-tstart:end))+0.02]);title('Demodulated signal2') 
figure;subplot(2,1,1); plot(t(end-tstart:end),msg3(end-tstart:end),'Linewidth',2);xlim([t(end-tstart) t(end)]);title('Message signal3');
subplot(2,1,2); plot(t(end-tstart:end),LIF2_HO3(end-tstart:end),'Linewidth',2);xlim([t(end-tstart) t(end)]);ylim([min(LIF2_HO3(end-tstart:end))-0.02 max(LIF2_HO3(end-tstart:end))+0.02]);title('Demodulated signal3')
figure;subplot(2,1,1); plot(t(end-tstart:end),msg4(end-tstart:end),'Linewidth',2);xlim([t(end-tstart) t(end)]);title('Message signal4');
subplot(2,1,2); plot(t(end-tstart:end),LIF2_HO4(end-tstart:end),'Linewidth',2);xlim([t(end-tstart) t(end)]);ylim([min(LIF2_HO4(end-tstart:end))-0.02 max(LIF2_HO4(end-tstart:end))+0.02]);title('Demodulated signal4') 