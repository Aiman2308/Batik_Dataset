function image = SURFKeypointVisualizer(Image, KeyPoints)
    % Creating color image
    image = cat(3, Image, Image, Image);

    for i = 1:length(KeyPoints)
        [x, y] = KeyPoints{i}.coordinates();
        circle = [y, x, 5];
        
        % Adding circles to keypoint locations
        image = insertShape(image, 'circle', circle, 'LineWidth', 1, 'color', [255, 0, 0], 'SmoothEdges', false);
    end

    for i = 1:length(KeyPoints)
        [x, y] = KeyPoints{i}.coordinates();
        dir = KeyPoints{i}.direction();
        line = [y, x, y + 10 * sind(dir), x + 10 * cosd(dir)];
        
        % Adding lines to keypoint locations
        image = insertShape(image, 'line', line, 'LineWidth', 1, 'color', [0, 0, 255]);
    end

    for i = 1:6:length(KeyPoints)
        [x, y] = KeyPoints{i}.coordinates();
        
        % Distinguishing keypoint location with green dots
        image(round(x), round(y), :) = [0, 255, 0];
    end
end
