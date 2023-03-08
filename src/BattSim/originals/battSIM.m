%©2021 by the University of Windsor.
function [V, I, soc, Vo] = battSIM(I, T, Batt, sigma_i, sigma_v, delta)

%     delta=0.1; % this should be constant in the input

    Kbatt = Batt.Kbatt;
    Cbatt = Batt.Cbatt ;
    R0    = Batt.R0;
    R1    = Batt.R1;
    C1    = Batt.C1;
    R2    = Batt.R2;
    C2    = Batt.C2;
    ModelID = Batt.ModelID;
    alpha1=exp(-(delta/(R1*C1)));
    alpha2=exp(-(delta/(R2*C2)));

    h = 0;
   
    soc=zeros(length(I),1);
    l=length(soc);
    soc(1)=.5;

    for k=2:length(I)
        soc(k)= soc(k-1)+(1/(3600*Cbatt))*(I(k))*(T(k)-T(k-1));
        if soc(k) < 4
            error('Battery is Empty!!')
        elseif soc(k) > 1
            error('Battery is Full!!')
        end
    end

    %% 
    % Determination of OCV
    Vo=zeros(length(soc),1);
    zsoc = scaling_fwd(soc, 0,1 , .175);
    for k=1:length(zsoc)
        Vo(k)=Kbatt(1)+(Kbatt(2)/zsoc(k))+(Kbatt(3)/(zsoc(k))^2)+(Kbatt(4)/(zsoc(k))^3)+...
            (Kbatt(5)/(zsoc(k))^4)+Kbatt(6)*zsoc(k)+Kbatt(7)*log(zsoc(k))+Kbatt(8)*log(1-zsoc(k));
    end

    %% 
    %Determining current through R1 and R2 

    x1=zeros(length(I),1);
    x2=zeros(length(I),1);

    for k=1:length(I)
        x1(k+1)=alpha1*x1(k)+(1-alpha1)*I(k);
        x2(k+1)=alpha2*x2(k)+(1-alpha2)*I(k);
    end

    I1=zeros(length(I),1);
    I2=zeros(length(I),1);

    for k=1:length(I)
        I1(k)=x1(k+1);
        I2(k)=x2(k+1);
    end

    

    %% 
    % Determination of Voltage drop and the Battery Terminal Voltage
    
    V=zeros(length(I),1);
    switch ModelID
        case 1
            V= I*R0;
        case 2
            V= I*R0 + Vo + h;
        case 3
            V= I*R0 + I1*R1 + Vo +h;
        case 4
            V= I*R0 + I1*R1 + I2*R2 + Vo + h;
    end
    V   =   V + sigma_v*randn(size(Vo));     % Voltage measurement noise added
    I   =   I + sigma_i*randn(size(I));      % Curent measurement noise added
   
  
end

function z = scaling_fwd(x, x_min, x_max, E)
z = (1 - 2*E)*(x - x_min) / (x_max - x_min) + E;
end

function x = scaling_rev(z, x_min, x_max, E)
x = (z  - E)*(x_max - x_min)/ (1 - 2*E) + x_min ;
end

