function err = build_decoder(enc,dataset,T)
if nargin == 2
    T = enc.T
end

enc.dec = dictionary([],{});
elem = {zeros(dataset.K,1), 0, -1};
% Elem structure is [frequency vector], total, class label

% Loop over the dataset and put points into rectangles.
for i=1:dataset.N
    % 1. Get codeword index for each data point
    c = 0;
    for j=1:T
        if dataset.X(i,enc.f(j)) > enc.t(j)
            c = c + 2^(j-1);
        end
    end
    % Look up rectangle data structure
    if isKey(enc.dec,c)
        e = enc.dec{c};
    else
        e = elem;
    end
    y = dataset.y(i) + 1; % Fix zero based to one based
    e{1}(y) = e{1}(y) + 1; % Increment this class
    e{2} = e{2} + 1; % Increment total
    enc.dec{c} = e;
end

% Fix the relative frequencies and set up the class label
elist = entries(enc.dec,'struct');
for i=1:length(elist)
    c = elist(i).Key;
    e = enc.dec{c};
    e{1} = e{1}/e{2};
    [~,maxind] = max(e{1});
    maxind = maxind - 1; % Fix one based to zero based
    e{3} = maxind; % Define the class label
    enc.dec{c} = e;
end

% Now calculate the error on the dataset
err = 0;
for i=1:dataset.N
% 1. Get codeword index for each data point
    c = 0;
    for j=1:T
        if dataset.X(i,enc.f(j)) > enc.t(j)
            c = c + 2^(j-1);
        end
    end
    % Look up rectangle data structure
    if isKey(enc.dec,c)
        e = enc.dec{c};
    else
        e = elem;
        fprintf("This should never happen.\n");
    end
    if dataset.y(i) ~= e{3}
        err = err + 1;
    end
end
err = err / dataset.N;
fprintf('Dictionary size = %d\n',numEntries(enc.dec));
