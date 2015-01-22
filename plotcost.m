%plots the cost function for a random instances object. Might take a while.
%Runing "matlabpool open" before calling this script is a good idea.

%you can change the granularity or the bounds here
granularity = 10;
bound1 = 2.5;
bound2 = 2.5;
costf = utils.CostFunction();

%create the meshgrid and initilizes the cost
x = -bound1:bound1/granularity*2:bound1;
y = -bound2:bound2/granularity*2:bound2;
[X,Y] = meshgrid(x,y);
n = length(x);
Z = zeros(n,n);

%generate the random dataset
generator = random_generators.RandomInstancesGenerator();
[dataset,weighter,alpha,true_w] = generator.generate();
%calculate the cost for each point
parfor i=1:n
    for k=1:n
        cost = dataset.calcCost(weighter,alpha,costf,[X(i,k)+0.000001,Y(i,k)+0.000001]);
        Z(i,k) = cost;
    end
end
figure;surf(X,Y,Z);xlabel('w1');ylabel('w2');title('cost');
figure;surf(X,Y,log(Z));xlabel('w1');ylabel('w2');title('log cost');