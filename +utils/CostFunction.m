classdef CostFunction < handle
    %Calculates the cost and gradient for a given delta value
    %the function is usually a sigmoid
    
    properties
        b = .00001;%b parameter present in every type of cost function refer to Leskovec
        type = 1;%1 WMW loss function
    end
    
    methods
        function costf = CostFunction(b,type)
           if nargin > 0
             costf.b = b;
             costf.type = type;
           end
        end
        
        function cost = calcCost(cf,x)
            if cf.type == 1
                cost = 1.0./(1.0+exp(-x./cf.b)); 
            end
        end
        
        function gradient = calcGradient(cf,x)
           if cf.type == 1 
               tmp = 1.0 ./ (1+exp(x./cf.b));
               gradient = tmp .* (1-tmp) ./ cf.b;
           end
        end
        
        function [cost,gradient] = calcCostAndGradient(cf,x)
           if cf.type == 1 
               cost = 1.0./(1.0+exp(-x./cf.b)); 
               gradient = cost .* (1-cost) ./ cf.b;
           end
        end
            
    end
    
end

