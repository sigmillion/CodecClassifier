function d = init_detector(M)
d.M = M;
d.m = 0;
d.G = zeros(2,M);
d.t = zeros(1,M);
d.alpha = zeros(1,M);

num = 30;
th = 2*pi*([0:num-1]+0.5)/num;
d.A = [cos(th);sin(th)];
d.num = num;
