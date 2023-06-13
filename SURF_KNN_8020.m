%% Load images and extract SURF features
train_dir = ("C:\Users\User\OneDrive - Universiti Teknikal Malaysia Melaka\Documents\MATLAB\SIFT\Training80");
test_dir = ("C:\Users\User\OneDrive - Universiti Teknikal Malaysia Melaka\Documents\MATLAB\SIFT\Testing20");

train_files = dir(fullfile(train_dir, '*.jpg')); % Get the list of JPEG files in the training directory
test_files = dir(fullfile(test_dir, '*.jpg')); % Get the list of JPEG files in the testing directory

% SURF parameters
num_octaves = 4;
num_scales = 5;
threshold = 51000;

%% Preallocate arrays for SURF descriptors and labels
num_train_images = numel(train_files);
num_test_images = numel(test_files);

num_train_descriptors = 0;
num_test_descriptors = 0;

tic % start timer

% Process training images
for i = 1:num_train_images
    % Load image
    image_file = fullfile(train_dir, train_files(i).name);
    img = imread(image_file);
    img = rgb2gray(img);
    img = double(img);
    keyPoints = SURF(img);
    img = SURFKeypointVisualizer(img,keyPoints);
    
    % Display the image with its label
    subplot(6, 10, i);
    imshow(uint8(img));
    title(sprintf('Label: %d', i));
end

% Process testing images
for i = 1:num_test_images
    % Load image
    image_file = fullfile(test_dir, test_files(i).name);
    img = imread(image_file);
    img_gray = rgb2gray(img);

    % Extract SURF features with specified parameters
    points = detectSURFFeatures(rgb2gray(img), 'NumOctaves', num_octaves, 'NumScaleLevels', num_scales, 'MetricThreshold', threshold);
    [features, ~] = extractFeatures(img_gray, points);

    num_test_descriptors = num_test_descriptors + size(features, 1);
end

elapsed_time = toc; % stop timer and calculate elapsed time
fprintf('SURF feature extraction time: %.2f seconds\n', elapsed_time);

%% Preallocate arrays for training and testing SURF descriptors and labels
X_train = zeros(num_train_descriptors, 64);
Y_train = zeros(num_train_descriptors, 1);

X_test = zeros(num_test_descriptors, 64);
Y_test = zeros(num_test_descriptors, 1);

%% Extract SURF features and add to feature matrix for training set
idx_train = 1;
for i = 1:num_train_images
    % Load image
    image_file = fullfile(train_dir, train_files(i).name);
    img = imread(image_file);
    img_gray = rgb2gray(img);

    % Extract SURF features with specified parameters
    points = detectSURFFeatures(rgb2gray(img), 'NumOctaves', num_octaves, 'NumScaleLevels', num_scales, 'MetricThreshold', threshold);
    [features, ~] = extractFeatures(img_gray, points);

    % Add descriptors to feature matrix and add label to label vector for training set
    num_descriptors_image = size(features, 1);
    X_train(idx_train:idx_train+num_descriptors_image-1, :) = features;
    Y_train(idx_train:idx_train+num_descriptors_image-1) = i;
    idx_train = idx_train + num_descriptors_image;
end

%% Extract SURF features and add to feature matrix for testing set
idx_test = 1;
for i = 1:num_test_images
    % Load image
    image_file = fullfile(test_dir, test_files(i).name);
    img = imread(image_file);
    img_gray = rgb2gray(img);

    % Extract SURF features with specified parameters
    points = detectSURFFeatures(rgb2gray(img), 'NumOctaves', num_octaves, 'NumScaleLevels', num_scales, 'MetricThreshold', threshold);
    [features, ~] = extractFeatures(img_gray, points);

    % Add descriptors to feature matrix and add label to label vector for testing set
    num_descriptors_image = size(features, 1);
    X_test(idx_test:idx_test+num_descriptors_image-1, :) = features;
    Y_test(idx_test:idx_test+num_descriptors_image-1) = i;
    idx_test = idx_test + num_descriptors_image;
end

%% Train KNN classifier
k = 2; % set the number of neighbors to consider
knn_model = fitcknn(X_train, Y_train, 'NumNeighbors', k);

%% Test KNN classifier
Y_pred = predict(knn_model, X_test);

%% Evaluate performance
confusion_mat = confusionmat(Y_test, Y_pred);
accuracy = sum(diag(confusion_mat))/sum(confusion_mat(:));
precision = diag(confusion_mat)./sum(confusion_mat, 1)';
recall = diag(confusion_mat)./sum(confusion_mat, 2);
fprintf('Accuracy: %.2f%%\n', accuracy*100);
fprintf('Precision: %.2f%%\n', nanmean(precision)*100);
fprintf('Recall: %.2f%%\n', nanmean(recall)*100);

% Display the classified images
figure;
for i = 1:num_test_images
    % Load image
    image_file = fullfile(test_dir, test_files(i).name);
    img = imread(image_file);

    % Display the image with its predicted label
    subplot(3, 5, i);
    imshow(img);
    title(sprintf('Predicted: %d', Y_pred(i)));
end
sgtitle('Classified Images');
