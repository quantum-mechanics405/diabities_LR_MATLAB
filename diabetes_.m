% Load or define the diabetes dataset
% Assuming X contains predictor variables and y contains the binary outcome
% Replace this with actual data loading if available

% Sample data for demonstration
data = readtable('diabetes.csv');
X=table2array(data(:,1:end-1));
y = table2array(data(:,end));
% Train logistic regression model
mdl = fitglm(X, y, 'Distribution', 'binomial', 'Link', 'logit');

% Display model summary
% disp(mdl);

% Make predictions on training data
X_ = X(:,:);

y_pred_prob = predict(mdl, X_);

% Convert probabilities to binary predictions (using a threshold of 0.5)
y_pred_binary = round(y_pred_prob);

% Calculate accuracy
accuracy = sum(y_pred_binary == y) / length(y);
disp(['Accuracy: ', num2str(accuracy)]);
