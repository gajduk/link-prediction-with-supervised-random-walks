classdef RandomInstancesGenerator < handle
    %Used to generate a random instances object for testing. Users can
    %specify how many instance objects to generate and the generator for
    %the graph objects.
    
    properties
        K = 10;%top K links are chosen as positive the rest as negative
        alpha = .3;%restart probability
        num_graphs = 3;%number of graphs to generate
        num_instances = 50;%number of instances (sDL groups)
        graph_generator = random_generators.RandomGraphGenerator();%a generator used to generate graphs
        weighter = utils.EdgeWeighter(1);%weighter function
        w = [1,-1];%feature parameters, length must be same as the number of features, otheriwse num of features will be changed
    end
    
    methods
        function this = RandomInstancesGenerator()
            %returns a generator object with default parameters
        end
        
        function [dataset,weighter,alpha,w] = generate(this)
            %generates the instances objects as specified by the properties
            graphs(1,this.num_graphs) = core.Graph();
            this.graph_generator.num_features = length(this.w);
            for i=1:this.num_graphs
               graphs(i) = this.graph_generator.generate();
            end
            instances(1,this.num_instances) = core.Instance();
            k = 1;
            for i=1:this.num_graphs
               num_start_nodes = this.num_instances/this.num_graphs;
               if i <= mod(this.num_instances,this.num_graphs)
                   num_start_nodes = num_start_nodes+1;
               end
               for s=1:num_start_nodes
                    instance = core.Instance();                    
                    instance.graph = graphs(i);
                    instance.source_node_index = s;
                    pagerank = instance.calcPagerank(this.weighter,this.alpha,this.w);
                    [~,Idxs] = sort(pagerank);
                    instance.positive_links = Idxs(graphs(i).num_nodes-this.K+1:graphs(i).num_nodes);
                    instance.negative_links = Idxs(1:graphs(i).num_nodes-this.K);
                    instances(k) = instance;
                    k = k+1;
               end
            end
            %group all sDL triplests with their respective graphs together in an
            %instances object
            dataset = core.Instances(instances);
            weighter = this.weighter;
            alpha = this.alpha;
            w = this.w;
        end
    end
    
end

