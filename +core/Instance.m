classdef Instance < handle
    %Instance holds information on positive and negative links for a source
    %node in a graph.
    
    properties
        source_node_index = 1; % source node
        positive_links = []; % positive nodes (links)
        negative_links = []; % negative nodes (links)
        graph@core.Graph; %a graph object that contains information on links and features
    end
    
    methods
        function this = Instance(source_node_index,positive_links,negative_links,graph)
            if nargin > 0
                this.source_node_index = source_node_index;
                this.positive_links = positive_links;
                this.negative_links = negative_links;
                this.graph = graph;
            end
        end
        
        
        
        function cost = calcCost(this,weighter,alpha,costf,w)
            %calculates the cost for the given parameters w as defined by
            %Leskovec. The wighter and cost functions need to be specified
            %as well as the restart probability (alpha).
            pagerank = calcPagerank(this,weighter,alpha,w);
            cost = 0;
            for di=1:length(this.positive_links)
                cost = cost+sum(costf.calcCost(pagerank(this.negative_links)-pagerank(this.positive_links(di))));
            end
        end
        
       function [cost,gradient] = calcCostAndGradient(this,weighter,alpha,costf,w)
           %calculates the cost and the gradient/ for the given parameters w as defined by
            %Leskovec. The wighter and cost functions need to be specified
            %as well as the restart probability (alpha).
            
           %%%%%%%%%%%%%%%%%%%%
           %   calculate the weigthed adjacency matrix
           %% %%%%%%%%%%%%%%%%%
           adjMat = this.graph.getWeightedAdjMatrix(weighter,w);
           %%%%%%%%%%%%%%%%%%%% 
           % precalculate the sums of rows in the adjMat
           %%%%%%%%%%%%%%%%%%%%
           sumRowsAdjMat = full(sum(adjMat,2));
           [i,j,v_adjMat] = find(adjMat);
           v_sum_fuv_w = sumRowsAdjMat(i);
           v_sum_fuv_w_squared = v_sum_fuv_w.^2;
           %%%%%%%%%%%%%%%%%%%%
           %calculate the transition probability matrix with respect to a starting node s
           %%%%%%%%%%%%%%%%%%%%
           Q = this.calcTransitionProbabilityMatrixForSourceSparse(this.graph.num_nodes,i,j,v_adjMat,this.source_node_index,alpha,v_sum_fuv_w);
           Qt = Q';
           %% 
           %calc pagerank
           %% 
           p = zeros(this.graph.num_nodes,100);
           p(:,1) = repmat(1.0/this.graph.num_nodes,this.graph.num_nodes,1);
           last_iter = 0;
           for iter=2:100    
                p(:,iter) = Qt*p(:,iter-1);
                if sum((p(:,iter)-p(:,iter-1)).^2) < 1e-12
                       last_iter = iter;
                       break; 
                end
                if iter == 99
                     'p didnt converge' 
                end
           end
           pagerank = p(:,last_iter);
           %% 
           %calc derivative, for every feature
           %% 
           d_p = cell(this.graph.num_features,1);
           for k=1:this.graph.num_features
               %init gradient
               d_p_t = zeros(this.graph.num_nodes,1);
               d_p_t_1 = zeros(this.graph.num_nodes,1);
               
               tic
               %calculate dQ #1
               dQ = this.TcalcdQM(weighter,w,alpha,k,adjMat,v_sum_fuv_w,v_sum_fuv_w_squared);
               
              % dQ1 = sDL.TcalcdQ(weighter,w,alpha,k,adjMat,sumRowsAdjMat);
               
               
               dQt = dQ';
               for iter=1:100   
                  d_p_t = Qt*d_p_t_1+dQt*p(:,min(iter,last_iter));
                  if sum((d_p_t_1-d_p_t).^2) < 1e-12
                       break; 
                  end
                  d_p_t_1 = d_p_t;
                  if iter == 99
                     'dp didnt converge' 
                  end
               end
               d_p{k} = d_p_t'; 
           end
           
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %calc cost and gradient
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           l = repmat(this.negative_links,1,length(this.positive_links));
           d = repmat(this.positive_links,1,length(this.negative_links))';
           gradient = zeros(1,this.graph.num_features);
           [costs,gradients] = costf.calcCostAndGradient(pagerank(l)-pagerank(d));
           cost = sum(sum(costs));
           for k=1:this.graph.num_features
               gradient(k) = sum(sum(gradients.*(d_p{k}(l)-d_p{k}(d))));
           end
           
        end
        
        function pagerank = calcPagerank(this,weighter,alpha,w)
            %calculates the pagerank for this instance and the specified
            %parameters, weighter function and restart probability.
            %%%%%%%%%%%%%%%
            %  calculate the weigthed adjacency matrix
            %%%%%%%%%%%%%%% 
            adjMat = this.graph.getWeightedAdjMatrix(weighter,w);
            %%%%%%%%%%%%%%%
            %  precalculate the sums of rows in the adjMat
            %%%%%%%%%%%%%%%
            sumRowsAdjMat = full(sum(adjMat,2));
            [i,j,v_adjMat] = find(adjMat);
            v_sum_fuv_w = sumRowsAdjMat(i);
            %%%%%%%%%%%%%%%
            %  calculate the transition probability matrix with respect to a
            %  starting node s
            %%%%%%%%%%%%%%%
            Q = this.calcTransitionProbabilityMatrixForSourceSparse(this.graph.num_nodes,i,j,v_adjMat,this.source_node_index,alpha,v_sum_fuv_w);
            Qt = Q';
            %%%%%%%%%%%%%%
            % calc the actual pagerank using the transition probability matrix
            %%%%%%%%%%%%%%
            pagerank = repmat(1.0/this.graph.num_nodes,this.graph.num_nodes,1);
            previous_pagerank = repmat(1.0/this.graph.num_nodes,this.graph.num_nodes,1);
            for iter=1:100    
                pagerank = Qt*previous_pagerank;
                pagerank = pagerank./sum(pagerank);
                if sum((pagerank-previous_pagerank).^2) < 1e-12
                       break; 
                end
                if iter == 99
                     w
                     'Error: pagerank didnt converge' 
                end
                previous_pagerank = pagerank;
            end
         end
    end
    
    methods (Access = private)
         
        
         function dQ = TcalcdQM(sDL,weighter,w,alpha,k,adjMat,v_sum_fuv_w,v_sum_fuv_w_squared)
           [i,j,v_adjMat] = find(adjMat);
           %calc  dufv_dw
           dfuv_dwk = sDL.graph.G.*weighter.calcGradient(w(k),sDL.graph.features{k});
           [~,~,v_dfuv_dwk] = find(dfuv_dwk);
           if length(v_dfuv_dwk) ~= length(v_adjMat);
               [~,~,v_dfuv_dwk] = find(sDL.graph.features{k});
           end
           %calc sum_dufv_dw 
           sum_dufv_dw = full(sum(dfuv_dwk,2));
           v_sum_dfuv_dwk = sum_dufv_dw(i);
           %precalc dQ
           res = (1-alpha).*(v_dfuv_dwk.*v_sum_fuv_w-v_adjMat.*v_sum_dfuv_dwk)./v_sum_fuv_w_squared;
           dQ = sparse(i,j,res,sDL.graph.num_nodes,sDL.graph.num_nodes);
        end
                
        function dQ = TcalcdQ(sDL,weighter,w,alpha,k,adjMat,sumRowsAdjMatRepmat)
           %depracated
           dQ = sparse(sDL.graph.num_nodes,sDL.graph.num_nodes);
           sum_dufv_dw = zeros(sDL.graph.num_nodes,1);
           for j=1:sDL.graph.num_nodes
               for u=1:sDL.graph.num_nodes
                   if sDL.graph.G(j,u) == 1
                      sum_dufv_dw(j) = sum_dufv_dw(j)+weighter.calcGradient(w(k),sDL.graph.features{k}(j,u));
                   end
               end
           end
           for j=1:sDL.graph.num_nodes
                for u=1:sDL.graph.num_nodes
                    if sDL.graph.G(j,u) == 1
                    	dQ(j,u) = (1-alpha)*(weighter.calcGradient(w(k),sDL.graph.features{k}(j,u))*sumRowsAdjMatRepmat(j)-adjMat(j,u)*sum_dufv_dw(j))/(sumRowsAdjMatRepmat(j)^2);
                    end
                end
           end
           dQ = dQ;
        end
        
        function TransProbMatrix = calcTransitionProbabilityMatrixForSourceSparse(this,n,i,j,v_adjMat,source_node,alpha,v_row_sums)
            adjMat = v_adjMat./v_row_sums;
            TransProbMatrix = sparse(i,j,(1-alpha)*adjMat,n,n);
            TransProbMatrix(:,source_node) = TransProbMatrix(:,source_node)+alpha;
            TransProbMatrix(sum(TransProbMatrix,2)==alpha,source_node) = 1;
        end
        
        function TransProbMatrix = calcTransitionProbabilityMatrixForSource(this,adjMat,source_node,alpha,row_sums)
            %Q = normalize adjMat to make it row stochastic
            %Q = (1-alpha)Q+alpha*1(v=s)
            %1(v=s) is a matrix with zeros except for the s column that contains ones
            adjMat = bsxfun(@rdivide,adjMat,row_sums);
            TransProbMatrix = (1-alpha)*adjMat;
            TransProbMatrix(:,source_node) = TransProbMatrix(:,source_node)+alpha;
        end
    end
    
end

