classdef Instances < handle
    %A set of sDL groups with their respective graphs
    %an Instances object can be a training or a test dataset for
    %link prediction
    
    properties
        instances;%an array of sDLGroups
    end
    
    properties (SetAccess = private)
        n = 0;
    end
    
    methods
        function iset = Instances(instances)
           %initializes the instances with a set of sDL groups
           iset.instances = instances;
           iset.n = length(instances);
        end
        
        function num_instances = getNumberOfInstances(this)
            %returns the number of instances in this dataset
            num_instances = this.n;
        end
        
        function [cost,gradient] = calcCostAndGradient(this,weighter,alpha,costf,w)
           % calculates the cost and gradient for this Instances using a
           % given cost function. Calls the calcCostAndGradient on each
           % instance in its instances array
           costs = zeros(this.n,1);
           gradients = zeros(this.n,length(w));
           for i=1:this.n
               [cost_i,gradient_i] = this.instances(i).calcCostAndGradient(weighter,alpha,costf,w);
               costs(i) = cost_i;
               gradients(i,:) = gradient_i;               
           end
           cost = sum(w.^2)+sum(costs);
           gradient = 2*w+sum(gradients);
        end
        
        function cost = calcCost(this,weighter,alpha,costf,w)
           %calculates the cost for this Instances using a given cost
           %function. Calls the calcCost on each
           % instance in its instances array
           costs = zeros(this.n,1);
           for i=1:this.n
               costs(i) = this.instances(i).calcCost(weighter,alpha,costf,w);   
           end
           cost = sum(w.^2)+sum(costs);
        end
        
    end
    
end

