% This script uses Mutual Information to guide
% the region splitting.
% Author: Jake Gunther
% Date: December 8, 2023

% Generate data in 2D
num = 100;
dataset.D = 2;
x = randn(num,2);
r = randn(num,1) + 4;
th = 2*pi*rand(num,1);
y = [r.*cos(th),r.*sin(th)];
dataset.X = [x;y];
dataset.y = [zeros(num,1);ones(num,1)]; % Classes 0 and 1
dataset.K = 2; % Number of classes
dataset.N = length(dataset.y);
dataset.xmax = max(dataset.X(:,1));
dataset.xmin = min(dataset.X(:,1));
dataset.ymax = max(dataset.X(:,2));
dataset.ymin = min(dataset.X(:,2));
clear x r th y num;

% Plot dataset
figure(30); clf;
for i=1:dataset.N
    if dataset.y(i) > 0
        plot(dataset.X(i,1),dataset.X(i,2),'r+'); hold on;
    else
        plot(dataset.X(i,1),dataset.X(i,2),'bs'); hold on;
    end
end
axis equal;

% Perform encoding by applying AdaBoost algorithm using stumps
enc.T = 30; % Number of classifiers
enc.t = zeros(enc.T,1); % Thresholds
enc.f = zeros(enc.T,1); % Feature indexes
enc.MI = zeros(enc.T,1); % Mutual information

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
        Nc = configureDictionary('double','double');
        Mj = configureDictionary('double','double');
        Ncj = configureDictionary('double','double');
        
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
            Ncval = lookup(Nc,c,FallbackValue=0);
            Ncval = Ncval + 1;
            Nc(c) = Ncval;
            %insert(Nc,c,Ncval);

            Mjval = lookup(Mj,y,Fallbackvalue=0);
            Mjval = Mjval + 1;
            Mj(y) = Mjval;
            %insert(Mj,y,Mjval);

            Ncjval = lookup(Ncj,cj,Fallbackvalue=0);
            Ncjval = Ncjval + 1;
            Ncj(cj) = Ncjval;
            %insert(Ncj,cj,Ncjval);
            disp([c y cj]);
        end
        fprintf('Initialization\n');
        fprintf('Nc sum = %d\n',sumvals(Nc));
        fprintf('Mj sum = %d\n',sumvals(Mj));
        fprintf('Ncj sum = %d\n',sumvals(Ncj));
        pause;
        
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
            %Ncval = lookup(Nc,c,FallbackValue=0);
            Ncval = Nc(c);
            Ncval = Ncval - 1;
            if Ncval <= 0
                remove(Nc,c);
            else
                Nc(c) = Ncval;
            end
            %insert(Nc,c,Ncval);

            %Mjval = lookup(Mj,y,Fallbackvalue=0);
            Mjval = Mj(y);
            Mjval = Mjval - 1;
            if Mjval <= 0
                remove(Mj,y);
            else
                Mj(y) = Mjval;
            end
            %insert(Mj,y,Mjval);

            %Ncjval = lookup(Ncj,cj,Fallbackvalue=0);
            Ncjval = Ncj(cj);
            Ncjval = Ncjval - 1;
            if Ncjval <= 0
                remove(Ncj,cj);
            else
                Ncj(cj) = Ncjval;
            end
            %insert(Ncj,cj,Ncjval);
            
            % 4. Append a 0 to the codeword
            c = c - 2^(i-1); % Zero-based index
            cj = c*dataset.K + y; % Zero-based index
            
            % 5. Index the dictionaries and increment the counts
            Ncval = lookup(Nc,c,FallbackValue=0);
            Ncval = Ncval + 1;
            Nc(c) = Ncval;
            %insert(Nc,c,Ncval);

            Mjval = lookup(Mj,y,Fallbackvalue=0);
            Mjval = Mjval + 1;
            Mj(y) = Mjval;
            %insert(Mj,y,Mjval);

            Ncjval = lookup(Ncj,cj,Fallbackvalue=0);
            Ncjval = Ncjval + 1;
            Ncj(cj) = Ncjval;
            %insert(Ncj,cj,Ncjval);
            
            % 6. Compute mutual information
            e = entries(Ncj,'struct');
            MI(s) = 0;
            edge_sum = 0;
            for ie = 1:length(e)
                cj = e(ie).Key;
                c = floor(cj/dataset.K);
                y = cj - c*dataset.K;
                Ncval = Nc(c); %lookup(Nc,c);
                Mjval = Mj(y); %lookup(Mj,y);
                Ncjval = Ncj(cj); %lookup(Ncj,cj);
                wc = Ncval/dataset.N;
                wj = Mjval/dataset.N;
                wcj = Ncjval/dataset.N;
                v = wcj / (wc*wj);
                g = v*log(v);
                if g>20 % Clip
                    g = 20; disp('CLIPPED');
                end
                MI(s) = MI(s) + wc*wj*g;
                edge_sum = edge_sum + Ncjval;
            end % Loop over graph edges ei
            fprintf('edge_sum = %d\n',edge_sum);
            
            if MI(s) > themax
                themax = MI(s);
                maxind = ind(s);
            end

            fprintf('s=%d =======================\n',s);
            fprintf('Nc sum = %d\n',sumvals(Nc));
            fprintf('Mj sum = %d\n',sumvals(Mj));
            fprintf('Ncj sum = %d\n',sumvals(Ncj));            
        end % Loop over splits s
        if themax > bestmax
            bestmax = themax;
            bestind = d;
            bestthresh = 0.5*(dataset.X(maxind,d) + dataset.X(maxind+1,d));
        end
        figure(20);
        plot(MI); hold on;
    end % Loop over dimensions d
    figure(20); hold off;
    
    % Build the weak learner
    enc.f(i) = bestind;
    enc.t(i) = bestthresh;

    figure(30); ax = axis;
    if enc.f(i) == 1
        plot(enc.t(i)*[1 1],ax(3:4),'LineWidth',3);
    else
        plot(ax(1:2),enc.t(i)*[1 1],'LineWidth',3);
    end
    pause;
end
figure(30); hold off;
