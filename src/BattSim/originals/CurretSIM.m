function [T, I] = CurretSIM(Iprofile, I1, I2, delta, T, D)

    switch Iprofile

        case 'staircase'

            delta = 5; % milliseconds
            Nsp = iter; % #of staircase pulses
            Ns = 5; % #of samples for each pulse
            Nb = 4; % #of blocks
            Imag = [-40 -80 -120 -160]; % same as the #of blocks

            T = 0:delta:delta * Nsp * Ns * Nb - 1;
            %         T        = T*10^-3;
            T = T';

            I = zeros(Nsp * Ns * Nb, 1);

            l = 1;

            for k = 1:Nb * Nsp

                I((k - 1) * Ns + 1:k * Ns) = Imag(l) * ones(Ns, 1);
                l = l + 1;

                if (l == Nb + 1)
                    l = 1;
                end

            end

            I = I * 10^ - 3;

        case 'deepdischarge'
            delta = 5; % milliseconds
            Ns = 10; % #of samples for each pulse
            Nb = 2;
            Imag = [0 -1000]; % same as the #of blocks
            T = 0:delta:delta * Ns * Nb - 1;
            T = T * 10^ - 3;
            T = T';
            l = 1;

            for k = 1:Nb
                I((k - 1) * Ns + 1:k * Ns) = Imag(l) * ones(Ns, 1);
                l = l + 1;

                if (l == Nb + 1)
                    l = 1;
                end

            end

            I = I * 10^ - 3;

        case 'rectangular'
            %         delt = delta;
            %         pulsewidth = T;
            %         pulsetotal = D;
            delta = 1000; % milliseconds
            Ns = 500; % #of samples for each pulse
            Nb = 2;
            Imag = [-1000 0]; % same as the #of blocks
            T = 0:delta:delta * Ns * Nb - 1;
            T = T * 10^ - 3;
            T = T';

            l = 1;

            for k = 1:Nb
                I((k - 1) * Ns + 1:k * Ns) = Imag(l) * ones(Ns, 1);
                l = l + 1;

                if (l == Nb + 1)
                    l = 1;
                end

            end

            I = I * 10^ - 3;
            I = I';

        case 'rectangularnew'
            %         delta =  Sampling delta time in seconds
            %         T = Pulse-Width in seconds
            %         D = Total time in seconds
            %         Id = current value
            Ns_pulse = T / delta; % Number of samples in one on-off pulse
            Np = floor(D / T); % Number of on-off pulses
            Nsp2 = Ns_pulse / 2; % Number of samples in each half of a pulse
            Nb = 2; % Number of blocks = on + off pulse
            Nt = D / delta; % Total number of samples
            Imag = [I1 I2]; % Current vector
            T = 0:delta:D; % Time vector
            T = T(2:end)';
            l = 1;
            I = Imag(1) * ones(length(T), 1);

            for k = 1:Nb * Np
                I((k - 1) * Nsp2 + 1:k * Nsp2) = Imag(l) * ones(Nsp2, 1);
                l = l + 1;

                if (l == Nb + 1)
                    l = 1;
                end

            end

            if (Np * Ns_pulse ~= Nt)
                Q = Nt - Np * Ns_pulse;
                I(k * Nsp2 + 1:k * Nsp2 + Q) = I(1:Q);
            end

            I = I * 10^ - 3;

        case 'UDDS'

        case 'simulated dynamic'
    end

    if nargin == 0
        close all
        figure; hold on; grid on;
        plot(1000 * T, I, '-*')
        xlabel('Time(ms)'); ylabel('Current (A)')
        set(gca, 'fontsize', 14)
    end

end
