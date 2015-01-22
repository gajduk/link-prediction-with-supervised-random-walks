%generates a random dataset with prespecified parameters, then uses a
%parameter learner to learn them.

%generate a random dataset, if you want to change the number of instances,
%features or the weighter function, change the properties of the generator
%before invoking generate
generator = random_generators.RandomInstancesGenerator();
[dataset,weighter,alpha,true_w] = generator.generate();
%use a WMW cost function; a time limit of 30 seconds and progress printing
learner = utils.ParameterLearner(weighter,alpha,utils.CostFunction(),30,true);
learned_w = learner.learn(dataset);
disp(['The true parameters are: ',num2str(true_w)])
disp(['The learned parameters are: ',num2str(learned_w)])
