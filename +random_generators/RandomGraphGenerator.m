classdef RandomGraphGenerator < handle
    % Generates a random graph built with the hybrid process.
    % With probability p_preferential a link chosen using preferential attachment
    % and with probability 1-p_preferenial a link is chose unifrmly at
    % random. Also generates num_features random features for each edge.
    
    properties
        num_nodes = 100;%number of nodes in each graph
        num_features = 2;%number of features in each graph
        start_nodes = 10;%starting number of nodes
        p_preferential = .8;%probability for preferential attachment
    end
    
    methods
        function g = generate(this)
            %generates a random graph (using preferential attachement with probability p_preferential),
            %starting with start_nodes fully connected nodes up to a total of num_nodes nodes. 
            %Also generates num_features random features for each edge taken from a normal distribution (0,1)
            G = zeros(this.num_nodes,this.num_nodes);
            G(1:this.start_nodes,1:this.start_nodes) = ones(this.start_nodes,this.start_nodes)-diag(ones(1,this.start_nodes));
            degrees = zeros(1,this.num_nodes);
            degrees(1:this.start_nodes) = repmat(this.start_nodes-1,this.start_nodes,1);
            total_sum_degrees = (this.start_nodes-1)*this.start_nodes;
            for k=this.start_nodes+1:this.num_nodes
                perm = randperm(k-1);
                perm_i = 1;
                for j=1:this.start_nodes
                    l = 0;
                    if rand() < this.p_preferential
                        l = perm(perm_i);
                        perm_i = perm_i+1;
                    else
                        flag = true;
                        while flag
                            p = rand();
                            for i=1:k
                                p = p-double(degrees(i))/total_sum_degrees;
                                if p<0
                                    l = i;
                                    break
                                end
                            end 
                            if G(l,k) ~= 1
                                flag = false;
                            end
                        end
                    end
                    degrees(l) = degrees(l)+1;
                    total_sum_degrees = total_sum_degrees+1;
                    G(l,k) = 1;
                    G(k,l) = 1;
                end
                degrees(k) = this.start_nodes;
                total_sum_degrees = total_sum_degrees+this.start_nodes;
            end
            G = sparse(G);
            psi = cell(this.num_features);
            for k=1:this.num_features
               psi{k} = sparse(randn(this.num_nodes).*G);
            end
            g = core.Graph(this.num_nodes,this.num_features,G,psi,true);
        end
    end
    
end

