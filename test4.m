% This script uses Mutual Information to guide
% the region splitting.
% Author: Jake Gunther
% Date: December 8, 2023

% Date: December 16, 2023
% Note: It is now time to expand this to MNIST.
% For efficiency's sake, I will want to swap
% out the sorting and searching approach and
% instead use a fixed set of bins.  Because MNIST
% is grayscale image data, I can use bins that
% separate out the grayscale values.  Since the
% pixels are 8-bit values, I could use 256 levels.
% Or I could quantize a bit more and use fewer levels
% based on the most significantn bits of the pixel
% value.  I will still need to sort the values, but
% I can sweep through the quantized splits faster
% than sweeping through all possible splits.
% Maybe start by adding binning to this code in 2D
% and make sure you get that working first.

% Date: December 21, 2023
% Note: This version tests 8-bit data and 256-levels
% in 2D as a step toward MNIST in 784D.

% Date: December 21, 2023
% Note: This version moves over to MNIST data.

% Load MNIST dataset
fx = fopen('/Users/Jake/Downloads/MNIST_ORG/train-images.idx3-ubyte','rb');
fy = fopen('/Users/Jake/Downloads/MNIST_ORG/train-labels.idx1-ubyte','rb');
fread(fx,16,'uint8'); % Skip the header
fread(fy,8,'uint8'); % Skip the header
x = fread(fx,inf,'uint8');
y = fread(fy,inf,'uint8');
fclose(fx);
fclose(fy);
dataset.X = reshape(x,[28*28,60000]).';
dataset.y = y;
dataset.D = 28*28; % Number of dimensions
dataset.K = 10; % Number of classes
dataset.N = 60000;
dataset.S = 0.5 + [0:254]; % Split candidates
clear x y;

% Reduce dataset for testing
N = 10;
dataset.X = dataset.X(1:N:end,:);
dataset.y = dataset.y(1:N:end);
dataset.N = dataset.N/N;

% Perform encoding by applying AdaBoost algorithm using stumps
enc.T = 30; % Number of classifiers
enc.t = zeros(enc.T,1); % Thresholds
enc.f = zeros(enc.T,1); % Feature indexes
enc.MI = zeros(enc.T,1); % Mutual information
enc.dec = dictionary([],{}); % Decoder
elem = {zeros(dataset.K,1), 0};

% Loop over weak classifiers
for i=1:enc.T
    % Loop over dimensions
    bestmax = -inf;
    bestind = 0;
    bestthresh = 0;
    for d=1:dataset.D % Loop over dimensions in the feature vector
        % Create empty dictionaries
        Nc = dictionary([],{});
        Mj = dictionary([],{});
        Ncj = dictionary([],{});
        
        % Initialize the dictionary counts:
        for s=1:dataset.N
            % 1. Get codeword index for each data point
            c = 0;
            for j=1:i-1
                if dataset.X(s,enc.f(j)) > enc.t(j)
                    c = c + 2^(j-1);
                end
            end

            % 2. Append a 1 to the codeword
            y = dataset.y(s); % Zero-based index
            c = c + 2^(i-1); % Zero-based index
            cj = c*dataset.K + y; % Zero-based index

            % 3. Index the dictionaries and increment counts
            if isKey(Nc,c)
                Ncval = Nc{c};
            else
                Ncval = elem;
            end
            Ncval{1}(y+1) = Ncval{1}(y+1) + 1;
            Ncval{2} = Ncval{2} + 1;
            Nc{c} = Ncval;

            if isKey(Mj,y)
                Mjval = Mj{y};
            else
                Mjval = elem;
            end
            Mjval{1}(y+1) = Mjval{1}(y+1) + 1;
            Mjval{2} = Mjval{2} + 1;
            Mj{y} = Mjval;

            if isKey(Ncj,cj)
                Ncjval = Ncj{cj};
            else
                Ncjval = elem;
            end
            Ncjval{1}(y+1) = Ncjval{1}(y+1) + 1;
            Ncjval{2} = Ncjval{2} + 1;
            Ncj{cj} = Ncjval;
        end
        %fprintf('Initialization\n');
        %fprintf('Nc sum = %d\n',sumvals(Nc));
        %fprintf('Mj sum = %d\n',sumvals(Mj));
        %fprintf('Ncj sum = %d\n',sumvals(Ncj));
        %pause;

        % Extract the data for this dimension and sort it
        [~,ind] = sort(dataset.X(:,d),'ascend'); % Sort

        % Try all the splits and compute the error assuming stumps
        themax = -inf;
        maxind = 0;
        MI = zeros(length(dataset.S),1);
        s1 = 1;
        for s0 = 1:length(dataset.S) % Loop over splits
            while s1 <= dataset.N && dataset.X(ind(s1),d) < dataset.S(s0)
                % 1. Get codeword for x(s)
                c = 0;
                for j=1:i-1
                    if dataset.X(ind(s1),enc.f(j)) > enc.t(j)
                        c = c + 2^(j-1);
                    end
                end
                
                % 2. Append a 1 to the codeword
                y = dataset.y(ind(s1)); % Zero-based index
                c = c + 2^(i-1); % Zero-based index
                cj = c*dataset.K + y; % Zero-based index
            
                % 3. Index the dictionaries and decrement the counts
                Ncval = Nc{c};
                Ncval{1}(y+1) = Ncval{1}(y+1) - 1;
                Ncval{2} = Ncval{2} - 1;
                if Ncval{2} <= 0
                    Nc = remove(Nc,c); %Nc{c} = [];
                else
                    Nc{c} = Ncval;
                end

                Mjval = Mj{y};
                Mjval{1}(y+1) = Mjval{1}(y+1) - 1;
                Mjval{2} = Mjval{2} - 1;
                if Mjval{2} <= 0
                    Mj = remove(Mj,y); %Mj{y} = [];
                else
                    Mj{y} = Mjval;
                end

                Ncjval = Ncj{cj};
                Ncjval{1}(y+1) = Ncjval{1}(y+1) - 1;
                Ncjval{2} = Ncjval{2} - 1;
                if Ncjval{2} <= 0
                    Ncj = remove(Ncj,cj); %Ncj{cj} = [];
                else
                    Ncj{cj} = Ncjval;
                end
            
                % 4. Append a 0 to the codeword
                c = c - 2^(i-1); % Zero-based index
                cj = c*dataset.K + y; % Zero-based index
            
                % 5. Index the dictionaries and increment the counts
                if isKey(Nc,c)
                    Ncval = Nc{c};
                else
                    Ncval = elem;
                end
                Ncval{1}(y+1) = Ncval{1}(y+1) + 1;
                Ncval{2} = Ncval{2} + 1;
                Nc{c} = Ncval;

                if isKey(Mj,y)
                    Mjval = Mj{y};
                else
                    Mjval = elem;
                end
                Mjval{1}(y+1) = Mjval{1}(y+1) + 1;
                Mjval{2} = Mjval{2} + 1;
                Mj{y} = Mjval;

                if isKey(Ncj,cj)
                    Ncjval = Ncj{cj};
                else
                    Ncjval = elem;
                end
                Ncjval{1}(y+1) = Ncjval{1}(y+1) + 1;
                Ncjval{2} = Ncjval{2} + 1;
                Ncj{cj} = Ncjval;        
                
                s1 = s1 + 1;
            end

            % 6. Compute mutual information
            e = entries(Ncj,'struct');
            MI(s0) = 0;
            for ie = 1:length(e)
                cj = e(ie).Key;
                c = floor(cj/dataset.K);
                y = cj - c*dataset.K;
                Ncval = Nc{c};
                Mjval = Mj{y};
                Ncjval = Ncj{cj};
                wc = Ncval{2}/dataset.N;
                wj = Mjval{2}/dataset.N;
                wcj = Ncjval{2}/dataset.N;
                v = wcj / (wc*wj);
                g = v*log(v);
                if g>1000 % Clip
                    g = 1000; disp('CLIPPED');
                end
                MI(s0) = MI(s0) + wc*wj*g;
            end % Loop over graph edges ei
            
            if MI(s0) > themax
                themax = MI(s0);
                thethresh = dataset.S(s0);
            end

            %fprintf('s=%d =======================\n',s0);
            %fprintf('Nc sum = %d\n',sumvals(Nc));
            %fprintf('Mj sum = %d\n',sumvals(Mj));
            %fprintf('Ncj sum = %d\n',sumvals(Ncj));
        end % Loop over splits s0
        if themax > bestmax
            bestmax = themax;
            bestind = d;
            bestthresh = thethresh;
        end
        %figure(20);
        %plot(MI); hold on;
    end % Loop over dimensions d
    %figure(20); hold off; drawnow;
    
    % Build the weak learner
    enc.f(i) = bestind;
    enc.t(i) = bestthresh;

    err = build_decoder(enc,dataset,i);
    fprintf("%3d: error rate = %f, x[%3d] <= %f\n",i,err,enc.f(i),enc.t(i));

    %pause;
end

