classdef Graph < handle
    %Holds information about the graph topology and edge features.
    %Only existing edges have features.
    
    properties
        num_nodes = 0; %number of nodes in the graph
        num_features = 0; %number fo features in the graph
        G = [[]]; %unweigthed directed adjacency matrix for the graph [n x n] matrix of 0s and 1s
        features = {}; % f x {n x n} feature matrix - each double value corresponds to a single feature type and a single edge
        isSparse = true;
    end
    
    methods
        function g = Graph(num_nodes,num_features,G,features,isSparse)
            %Populates and returns a
            %Graph object with the supplied properties, 
            if nargin == 5
                g.num_nodes = num_nodes;
                g.num_features = num_features;
                g.G = G;
                g.features = features;   
                g.isSparse = isSparse;
            end
        end
        
        function adjMat = getWeightedAdjMatrix(g,weigther,w)
            %uses the weighter function and w parameters to combine the
            %features of each edge into a single double value (weight or
            %strength), returns aa n x n matrix of doubles
            if g.isSparse
                dot_product = sparse(g.num_nodes,g.num_nodes);
            else
                dot_product = zeros(g.num_nodes);
            end
            for k=1:g.num_features
                dot_product = dot_product + g.features{k}*w(k);
            end
            adjMat = g.G.*weigther.calcWeights(dot_product);
        end
        
        function setSparse(g,isSparse)
            %if tru forces the graph to use the sparse representation, and
            %the dense representation otherwise. Cost computation can
            %differ signicantly depending on graph representation. If your
            %graph is really sparse <5% od the edges exist the sparse
            %representation is best.
           if g.isSparse ~= isSparse
              if isSparse
                   g.G = sparse(g.G);
                   for k=1:g.num_features
                      g.features{k} = sparse(g.features{k});
                   end
              else
                  g.G = full(g.G); 
                  for k=1:g.num_features
                      g.features{k} = full(g.features{k});
                  end
              end
           end
        end
    end
    methods (Static)
        
    end
end

