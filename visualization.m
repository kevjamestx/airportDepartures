data = load('data.mat');
eventLog = data.EventLog;

scheduledTakeoffTime = datetime(eventLog(:, 1), 'ConvertFrom', 'posixtime'); % Convert first timestamp column
