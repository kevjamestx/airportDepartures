% Parameters
T_max = 180; % Maximum simulation time in minutes
F = 10; % Number of flights (adjustable)
gamma = 1; % Discount factor if applicable (not needed for minimax)
taxi_distribution = @(x) exprnd(x); % Erlang approximation for taxi time

% State Space Initialization
V = inf(T_max+1, F, T_max+1); % DP table for storing min max-delay values
policy = zeros(T_max+1, F); % To store optimal actions

% Terminal Condition: At final decision point, max delay is just the last takeoff delay
for f = 1:F
    for t_ready = 1:T_max
        V(T_max+1, f, :) = 0; % No further decisions
    end
end

% Backward Induction
for t = T_max:-1:1
    for f = 1:F
        % Get flights ready for pushback
        if is_ready(f, t) % Function that checks if flight f is ready at t
            
            % Possible actions: pushback at time t
            for t_pushback = t:T_max
                
                % Simulate transition: Get expected taxi-out time
                taxi_time = taxi_distribution(mean_taxi_time(f)); 
                t_wheels_up = t_pushback + taxi_time;
                
                % Compute new max delay
                max_delay_new = max(V(t+1, :, :), t_wheels_up - t);

                % Minimize over possible actions
                if max_delay_new < V(t, f, t_pushback)
                    V(t, f, t_pushback) = max_delay_new;
                    policy(t, f) = t_pushback; % Store best decision
                end
            end
        end
    end
end

% Extract optimal policy
optimal_policy = policy(1:T_max, :);

% Display results
disp('Optimal Pushback Schedule:');
disp(optimal_policy);