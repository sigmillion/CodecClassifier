% Generate data in 2D
num = 100;
dataset.D = 2;
x = randn(num,2);
r = randn(num,1) + 4;
th = 2*pi*rand(num,1);
y = [r.*cos(th),r.*sin(th)];
dataset.X = [x;y];
dataset.y = [-ones(num,1);+ones(num,1)];
dataset.N = length(dataset.y);
dataset.xmax = max(dataset.X(:,1));
dataset.xmin = min(dataset.X(:,1));
dataset.ymax = max(dataset.X(:,2));
dataset.ymin = min(dataset.X(:,2));
clear x r th y num;

figure(10); clf;

% Plot dataset
%plot_data(dataset);
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
enc.v = zeros(enc.T,1); % Left leaf value either -1 or +1
enc.a = zeros(enc.T,1); % Weak learner weights
enc.w = ones(dataset.N,1)/dataset.N; % Weights

figure(20); clf;
plot(enc.w); hold on;

% Loop over weak classifiers
for i=1:enc.T
    % Loop over dimensions
    bestmin = inf;
    bestind = 0;
    bestthresh = 0;
    for d=1:dataset.D
        % Extract the data for this dimension and sort it
        x = dataset.X(:,d); % Data
        [x,ind] = sort(x,'ascend'); % Sort
        y = dataset.y(ind); % Targets
        w = enc.w(ind); % Weights

        % Try all the splits and compute the error assuming stumps
        themin = inf;
        minind = 0;

        nlefta = 0; % Number of errors to the left of the splitpoint
        nrighta = sum(w.*(y ~= +1)); % Number of errors to the right of the split point
        errora = zeros(dataset.N-1,1);

        nleftb = 0;
        nrightb = sum(w.*(y ~= -1));
        errorb = zeros(dataset.N-1,1);

        for s = 1:dataset.N-1 % Loop over splits
            nlefta = nlefta + w(s)*(y(s) ~= -1);
            nrighta = nrighta - w(s)*(y(s) ~= +1);
            errora(s) = nlefta + nrighta;

            if errora(s) < themin
                themin = errora(s);
                thesense = -1;
                minind = s;
            end

            nleftb = nleftb + w(s)*(y(s) ~= +1);
            nrightb = nrightb - w(s)*(y(s) ~= -1);
            errorb(s) = nleftb + nrightb;

            if errorb(s) < themin
                themin = errorb(s);
                thesense = +1;
                minind = s;
            end
        end
        if themin < bestmin
            bestmin = themin;
            bestind = d;
            bestthresh = 0.5*(x(minind) + x(minind+1));
            bestsense = thesense;
        end
        
        % Plotting
        figure(10);
        plot(errora); hold on;
        plot(errorb,'--'); hold on;
    end
    figure(10);
    hold off;
    
    % Build the weak learner
    enc.f(i) = bestind;
    enc.t(i) = bestthresh;
    enc.v(i) = bestsense; % Left leaf value
    enc.a(i) = 0.5*log((1-bestmin)/bestmin);

    % Update the weights
    logic = dataset.X(:,enc.f(i)) <= enc.t(i);
    y = ones(dataset.N,1);
    y(logic) = enc.v(i);
    y(~logic) = -enc.v(i);
    w = enc.w .* exp(-enc.a(i)*(dataset.y .* y));
    enc.w = w./sum(w);

    % What is the error rate up to this point?
    H = zeros(dataset.N,1);
    for j=1:i
        logic = dataset.X(:,enc.f(i)) <= enc.t(i);
        y = ones(dataset.N,1);
        y(logic) = enc.v(i);
        y(~logic) = -enc.v(i);
        H = H + enc.a(j)*y;
    end
    H = sign(H);
    err_rate = sum(H ~= dataset.y)/dataset.N;
    fprintf("Iteration = %2d, error rate = %f\n",i,err_rate);
    
    figure(20);
    plot(enc.w); hold on;
    figure(30); ax = axis;
    if enc.f(i) == 1
        if enc.v(i) == -1
            plot(enc.t(i)*[1 1],ax(3:4),'LineWidth',3);
        else
            plot(enc.t(i)*[1 1],ax(3:4),'--','LineWidth',3);
        end
    else
        if enc.v(i) == -1
            plot(ax(1:2),enc.t(i)*[1 1],'LineWidth',3);
        else
            plot(ax(1:2),enc.t(i)*[1 1],'--','LineWidth',3);
        end
    end
    pause;
end
figure(30); hold off;
figure(20); hold off;
figure(10); hold off;