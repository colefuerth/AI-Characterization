
% dbstop('error')

k0           = -9.082;
k1           = 103.087;
k2           = -18.185;
k3           = 2.062;
k4           = -0.102;
k5           = -76.604;
k6           = 141.199;
k7           = -1.117;

Kbatt        = [k0; k1; k2; k3; k4; k5; k6; k7];

batt_model   = 3; % (1,2,3,4) batteyr is set to this model

Batt.Kbatt   = Kbatt;
Batt.Cbatt   = 1.9;
Batt.R0      = 0.2;
Batt.R1      = .1;
Batt.C1      = 5;
Batt.R2      = 0.3;
Batt.C2      = 500;
Batt.ModelID = batt_model;

%current simulation parameters
delta = 100*10^(-3);
Tc  = 10;  % sampling
D   = 100; % duration of the simulation
Id = -500; % amplitude of current (mA)

Batt.alpha1 = exp(-(delta/(Batt.R1*Batt.C1)));
Batt.alpha2 = exp(-(delta/(Batt.R2*Batt.C2)));

%current simulation
[T,I] = CurretSIM('rectangularnew',-Id, Id, delta, Tc, D);

%SNR and noise parameters
SNR     = 50;
sigma_i = 0; % current measurement noise
sigma_v = 10^(-SNR/20); % voltage measurement noise

% send current into the batery simulater
[Vbatt,Ibatt, ~, Vo] = battSIM(I, T, Batt, sigma_i, sigma_v, delta);

%current plotting
hI=figure; hold on; grid on; box on
subplot(211); box on; grid on
plot(T, Ibatt, 'linewidth', 2);
xlabel('Time (sec)') ; ylabel('Current (mA)')
grid on
subplot(212); box on; grid on
plot(T, Vbatt, 'linewidth', 2);
grid on
xlabel('Time (sec)') ; ylabel('Voltage (V)')






