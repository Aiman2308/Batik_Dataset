function Descriptors = SIFT(inputImage)
    % Convert inputImage to grayscale if needed
    
    % Detect SIFT features
    siftPoints = detectSIFTFeatures(inputImage);
    
    % Extract SIFT descriptors
    [features, validPoints] = extractFeatures(inputImage, siftPoints);
    
    % Create Descriptors object to store keypoint information
    Descriptors = cell(1, validPoints.Count);
    
    % Store keypoints and descriptors in Descriptors object
    for i = 1:validPoints.Count
        kp = KeyPoint;
        kp.Coordinates = validPoints.Location(i, :);
        kp.Scale = validPoints.Scale(i);
        kp.Magnitude = validPoints.Metric(i);
        kp.Direction = validPoints.Orientation(i);
        kp.Descriptor = features(i, :);
        
        Descriptors{i} = kp;
    end
end
