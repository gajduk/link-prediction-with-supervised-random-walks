classdef EdgeWeighter < handle
    
    properties
       type = 1;%one is exponential edge strength, 2 is logistic edge strength 
    end
       
    methods
        function weighter = EdgeWeighter(type)
            if nargin > 0
                weighter.type = type;                 
            end
        end
        
        function weights = calcWeights(weighter,dot_product)
            if weighter.type == 1
                weights = spfun(@exp,dot_product);
            else
                weights = 1/(1+exp(-dot_product));
            end
        end
        
        function gradient = calcGradient(weighter,w,psi)
           if weighter.type == 1
                gradient = psi.*spfun(@exp,psi*w);
            else
                gradient = psi.*exp(-psi*w)./(1+exp(-psi*w)).^2;
            end 
        end
    end
    
end

