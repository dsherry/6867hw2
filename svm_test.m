function svm_test(name)
disp('======Training======');
% load data from csv files

data = importdata(strcat('data/data_',name,'_train.csv'));

X = data(:,1:2);
Y = data(:,3);

% Carry out training, primal and/or dual

% Define the predictSVM(x) function, which uses trained parameters

hold on;

% plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], 'SVM Train');

disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data_',name,'_validate.csv'));
X = validate(:,1:2);
Y = validate(:,3);

% plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], 'SVM Validate');

