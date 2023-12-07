function dataset = gen_data(num)
dataset.D = 2;
dataset.N = num;

x = randn(num,2);
r = randn(num,1) + 4;
th = 2*pi*rand(num,1);
y = [r.*cos(th);r.*sin(th)];
dataset.X = [x;y];
dataset.y = [-ones(num,1),+ones(num,1)];
dataset.N = length(dataset.y);
dataset.w = (1/dataset.N)*ones(dataset.N,1);
dataset.xmax = max(dataset.X(1,:));
dataset.xmin = min(dataset.X(1,:));
dataset.ymax = max(dataset.X(2,:));
dataset.ymin = min(dataset.X(2,:));
