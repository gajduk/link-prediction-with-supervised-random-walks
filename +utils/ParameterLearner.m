classdef ParameterLearner
    %A utility class that laerns the best parameters for a specifeid
    %instances with the learn method. Groups togehter information about the
    %wighter and cost functions and the restart probability. Can also
    %define the maximal time limit for learning the parameters.
    
    properties
        weighter@utils.EdgeWeighter;%wegihter function
        costf@utils.CostFunction;%cost function
        alpha = .3;%restart probability
        time_limit = 30;%time limit for optimization in seconds
        print_progress=true;%whether or not to display progress while learning the parameters
    end
    
    methods
        function this = ParameterLearner(weighter,alpha,costf,time_limit,print_progress)
            %creates a new parameter learner either with all parameters
            %specified or with default values.
            if nargin == 5
               this.weighter = weighter;
               this.costf = costf;
               this.alpha = alpha;
               this.time_limit = time_limit;
               this.print_progress = print_progress;
            else
               this.weighter = utils.EdgeWeighter();
               this.costf = utils.CostFunction();
            end
        end

        function w = learn(this,instances)
            %learns the parameters using the simulated anealing method
            %the parameters are bounded to -3,3
            nf = instances.instances(1).graph.num_features;
            w0 = repmat(.000001,1,nf);
            ObjectiveFunction = @(x) instances.calcCost(this.weighter,this.alpha,this.costf,x);
            display = 'iter';
            if ~this.print_progress
                display = 'off';
            end
            options = saoptimset('Display',display,'ReannealInterval',10,...
                                'ObjectiveLimit',nf+1,'TimeLimit',this.time_limit,...
                                'TemperatureFcn',@temperatureboltz,'TolFun',1e-10);
            w = simulannealbnd(ObjectiveFunction,w0,-repmat(3.0,1,nf),repmat(3.0,1,nf),options);
        end
    end
    
end

