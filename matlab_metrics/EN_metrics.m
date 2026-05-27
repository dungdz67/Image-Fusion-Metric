% Reference from https://github.com/Linfeng-Tang/Image-Fusion General Metrics

function ent = EN_metrics(image)
    % Calculate entropy of a single image
    % Input: image - grayscale or color image
    % Output: ent - entropy value
    
    % Convert to grayscale if color image
    if size(image, 3) == 3
        image = rgb2gray(image);
    end
    
    % Normalize image to 0-255 range if necessary
    if max(image(:)) <= 1
        image = uint8(image * 255);
    else
        image = uint8(image);
    end
    
    % Calculate histogram
    hist = imhist(image);
    
    % Normalize histogram to get probability distribution
    hist_norm = hist / sum(hist);
    
    % Remove zero values to avoid log(0)
    hist_norm = hist_norm(hist_norm > 0);
    
    % Calculate entropy: H = -sum(p * log2(p))
    ent = -sum(hist_norm .* log2(hist_norm));
end