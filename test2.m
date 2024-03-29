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

% Generate data in 2D
num = 1000;
dataset.D = 2;
x = randn(num,2);
r = randn(num,1) + 4;
th = 2*pi*rand(num,1);
y = [r.*cos(th),r.*sin(th)];
dataset.X = [x;y];
r = randn(num,1) + 8;
th = 2*pi*rand(num,1);
y = [r.*cos(th),r.*sin(th)];
dataset.X = [dataset.X;y];

%dataset.y = [zeros(num,1);ones(num,1)]; % Classes 0 and 1
dataset.y = [zeros(num,1);ones(num,1);2*ones(num,1)]; % Classes 0 and 1
dataset.K = 3; % Number of classes
dataset.N = length(dataset.y);
dataset.xmax = max(dataset.X(:,1));
dataset.xmin = min(dataset.X(:,1));
dataset.ymax = max(dataset.X(:,2));
dataset.ymin = min(dataset.X(:,2));
clear x r th y num;

% Plot dataset
figure(30); clf;
for i=1:dataset.N
    if dataset.y(i) == 0
        plot(dataset.X(i,1),dataset.X(i,2),'r+'); hold on;
    elseif dataset.y(i) == 1
        plot(dataset.X(i,1),dataset.X(i,2),'bs'); hold on;
    else
        plot(dataset.X(i,1),dataset.X(i,2),'k^'); hold on;
    end
end
axis equal;

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
        % Extract the data for this dimension and sort it
        x = dataset.X(:,d); % Data
        [~,ind] = sort(x,'ascend'); % Sort

        % Try all the splits and compute the error assuming stumps
        themax = -inf;
        maxind = 0;

        % Create empty dictionaries
        if 0
            Nc = configureDictionary('double','double');
            Mj = configureDictionary('double','double');
            Ncj = configureDictionary('double','double');
        else
            Nc = dictionary([],[]);
            Mj = dictionary([],[]);
            Ncj = dictionary([],[]);
        end
        
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
                Ncval = Nc(c);
                Ncval = Ncval + 1;
            else
                Ncval = 1;
            end
            Nc(c) = Ncval;

            if isKey(Mj,y)
                Mjval = Mj(y);
                Mjval = Mjval + 1;
            else
                Mjval = 1;
            end
            Mj(y) = Mjval;

            if isKey(Ncj,cj)
                Ncjval = Ncj(cj);
                Ncjval = Ncjval + 1;
            else
                Ncjval = 1;
            end
            Ncj(cj) = Ncjval;
        end
        %fprintf('Initialization\n');
        %fprintf('Nc sum = %d\n',sumvals(Nc));
        %fprintf('Mj sum = %d\n',sumvals(Mj));
        %fprintf('Ncj sum = %d\n',sumvals(Ncj));
        %pause;
        
        MI = zeros(dataset.N-1,1);
        for s = 1:dataset.N-1 % Loop over splits
            % 1. Get codeword for x(s)
            c = 0;
            for j=1:i-1
                if dataset.X(ind(s),enc.f(j)) > enc.t(j)
                    c = c + 2^(j-1);
                end
            end
            
            % 2. Append a 1 to the codeword
            y = dataset.y(ind(s)); % Zero-based index
            c = c + 2^(i-1); % Zero-based index
            cj = c*dataset.K + y; % Zero-based index
            
            % 3. Index the dictionaries and decrement the counts
            Ncval = Nc(c);
            Ncval = Ncval - 1;
            if Ncval <= 0
                Nc(c) = [];
            else
                Nc(c) = Ncval;
            end

            Mjval = Mj(y);
            Mjval = Mjval - 1;
            if Mjval <= 0
                Mj(y) = [];
            else
                Mj(y) = Mjval;
            end

            Ncjval = Ncj(cj);
            Ncjval = Ncjval - 1;
            if Ncjval <= 0
                Ncj(cj) = [];
            else
                Ncj(cj) = Ncjval;
            end
            
            % 4. Append a 0 to the codeword
            c = c - 2^(i-1); % Zero-based index
            cj = c*dataset.K + y; % Zero-based index
            
            % 5. Index the dictionaries and increment the counts
            if isKey(Nc,c)
                Ncval = Nc(c);
                Ncval = Ncval + 1;
            else
                Ncval = 1;
            end
            Nc(c) = Ncval;

            if isKey(Mj,y)
                Mjval = Mj(y);
                Mjval = Mjval + 1;
            else
                Mjval = 1;
            end
            Mj(y) = Mjval;

            if isKey(Ncj,cj)
                Ncjval = Ncj(cj);
                Ncjval = Ncjval + 1;
            else
                Ncjval = 1;
            end
            Ncj(cj) = Ncjval;            
            % 6. Compute mutual information
            e = entries(Ncj,'struct');
            MI(s) = 0;
            for ie = 1:length(e)
                cj = e(ie).Key;
                c = floor(cj/dataset.K);
                y = cj - c*dataset.K;
                Ncval = Nc(c);
                Mjval = Mj(y);
                Ncjval = Ncj(cj);
                wc = Ncval/dataset.N;
                wj = Mjval/dataset.N;
                wcj = Ncjval/dataset.N;
                v = wcj / (wc*wj);
                g = v*log(v);
                if g>20 % Clip
                    g = 20; disp('CLIPPED');
                end
                MI(s) = MI(s) + wc*wj*g;
            end % Loop over graph edges ei
            
            if MI(s) > themax
                themax = MI(s);
                maxind = ind(s);
                maxind1 = ind(s+1);
            end

            %fprintf('s=%d =======================\n',s);
            %fprintf('Nc sum = %d\n',sumvals(Nc));
            %fprintf('Mj sum = %d\n',sumvals(Mj));
            %fprintf('Ncj sum = %d\n',sumvals(Ncj));
        end % Loop over splits s
        if themax > bestmax
            bestmax = themax;
            bestind = d;
            bestthresh = 0.5*(dataset.X(maxind,d) + dataset.X(maxind1,d));
        end
        figure(20);
        plot(MI); hold on;
    end % Loop over dimensions d
    figure(20); hold off;
    
    % Build the weak learner
    enc.f(i) = bestind;
    enc.t(i) = bestthresh;

    err = build_decoder(enc,dataset,i);
    fprintf("%3d: error rate = %f\n",i,err);

    figure(30); ax = axis;
    if enc.f(i) == 1
        plot(enc.t(i)*[1 1],ax(3:4),'LineWidth',3);
    else
        plot(ax(1:2),enc.t(i)*[1 1],'LineWidth',3);
    end
    pause;
end
figure(30); hold off;
