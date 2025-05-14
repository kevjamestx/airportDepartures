% Flights defined by
% (Flt Num, Scheduled Pushback Time, Pushback Delay)
f1 = [1,0,5];
f2 = [2,1,3];

%Input States
F = [f1, f2];
ts = 20; % Total time

% Paramaters
%Taxi Time
Tt = 5; % Deterministic for now

% Base cases
T(x2)

% Subproblem
T = zeros(length(F), ts);

% Recursion
