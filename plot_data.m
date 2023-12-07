function plot_data(dataset)
c2 = [     0    0.4470    0.7410];
c1 = [0.8500    0.3250    0.0980];

ind = find(dataset.y > 0);
plot(dataset.X(1,ind),dataset.X(2,ind),'.','Color',c1);
hold on;
ind = find(dataset.y < 0);
plot(dataset.X(1,ind),dataset.X(2,ind),'.','Color',c2);
hold off;
t = max(abs([dataset.xmax,dataset.xmin,...
             dataset.ymax,dataset.ymin]));
axis(t*[-1 1 -1 1]);
axis square;
