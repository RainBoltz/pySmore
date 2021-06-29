from pysmore.libs.util import graph_reader_generator
import numpy as np
from collections import deque
from tqdm import tqdm

class Graph:
    def __init__(self, path, delimiter='\t', mode='node', undirected=False):
        self._rng_generator = np.random.default_rng()   # random generator
        self._sampled_vertices = False                  # cached sampling (list)
        self._sampled_edges = False                     # cached edge sampling (list)

        self._graph = None                              # main graph
        self._vertices = []                             # vertex name list
        self._vertex_idxs = {}                          # vertex index mapper (from name list)
        self._contexts = []                             # context name list
        self._context_idxs = {}                         # context index mapper (from name list)
        self._vertex_count = 0                          # vertex amount
        self._context_count = 0                         # context amount
        self._edge_count = 0                            # edge amount

        if mode == 'node':
            self._create_graph_by_nodes(path, delimiter, undirected)
        elif mode == 'edge':
            self._create_graph_by_edges(path, delimiter, undirected)
        else:
            raise 'unknown graph type'
    
    def _create_graph_by_nodes(self, path, delimiter, undirected):
        self._graph = {}
        for vertex, context, weight in tqdm(
            graph_reader_generator(path=path, delimiter=delimiter),
            desc='reading graph (node based) '):
            
            self._vertices.append(vertex)
            self._contexts.append(context)
            if vertex not in self._graph:
                self._graph[vertex] = {}
            self._graph[vertex][context] = weight
            self._edge_count += 1

            if undirected:
                if context not in self._graph:
                    self._graph[context] = {}
                self._graph[context][vertex] = weight
                self._edge_count += 1

        if undirected:
            self._vertices = list(self._graph)
            self._contexts = self._vertices

            self._vertex_idxs = { v: idx for idx, v in enumerate(self._vertices) }
            self._context_idxs = self._vertex_idxs

            self._vertex_count = len(self._vertices)
            self._context_count = self._vertex_count
        
        else:
            self._vertices = list(set(self._vertices))
            self._contexts = list(set(self._contexts))
            
            self._vertex_idxs = { v: idx for idx, v in enumerate(self._vertices) }
            self._context_idxs = { c: idx for idx, c in enumerate(self._contexts) }

            self._vertex_count = len(self._vertices)
            self._context_count = len(self._contexts)

    
    def _create_graph_by_edges(self, path, delimiter, undirected):
        self._graph = []
        for vertex, context, weight in tqdm(
            graph_reader_generator(path=path, delimiter=delimiter),
            desc='reading graph (edge based) '):
            
            self._vertices.append(vertex)
            self._contexts.append(context)

            self._graph.append([vertex, -1, context, -1, weight])
            if undirected:
                self._graph.append([context, -1, vertex, -1, weight])            

        self._vertices = list(set(self._vertices))
        self._contexts = list(set(self._contexts))

        self._vertex_count = len(self._vertices)
        self._context_count = len(self._contexts)

        if undirected:
            self._vertices += self._contexts
            self._contexts = self._vertices

            self._vertex_count += self._context_count 
            self._context_count = self._vertex_count

            self._vertex_idxs = { v: idx for idx, v in enumerate(self._vertices) }
            self._context_idxs = self._vertex_idxs
        else:
            self._vertex_idxs = { v: idx for idx, v in enumerate(self._vertices) }
            self._context_idxs = { c: idx for idx, c in enumerate(self._contexts) }
            
        self._edge_count = len(self._graph)

        for i in range(self._edge_count):
            self._graph[i][1] = self._vertex_idxs[self._graph[i][0]]
            self._graph[i][3] = self._context_idxs[self._graph[i][2]]
    
    @property
    def vertex_count(self):
        return self._vertex_count

    @property
    def context_count(self):
        return self._context_count

    @property
    def edge_count(self):
        return self._edge_count
    
    @property
    def vertices(self):
        return self._vertices

    @property
    def contexts(self):
        return self._contexts

    
    def initialize_random_state(self, seed=None):
        if not seed:
            import time
            seed = int(time.time())
        self._rng_generator = np.random.default_rng(seed)


    # weight getters
    def get_weight_from_vertex_context(self, vertex: str, context: str):
        return self._graph[vertex][context]

    def get_weight_from_edge_idx(self, edge_idx: int):
        return self._graph[edge_idx][-1]

    
    # index getters
    def get_vertices_index(self, vertices: list):
        return [ self._vertex_idxs[vertex] for vertex in vertices ]

    def get_contexts_index(self, contexts: list):
        return [ self._context_idxs[context] for context in contexts ]

    # cached-sample functions
    def cache_vertex_samples(self, amount):
        sampled_vertices = self._rng_generator.integers(low=0, high=self._vertex_count, size=amount)
        self._sampled_vertices = deque(sampled_vertices)

    def cache_edge_samples(self, amount):
        sampled_edges = self._rng_generator.integers(low=0, high=self._edge_count, size=amount)
        self._sampled_edges = deque(sampled_edges)

    def draw_a_vertex_from_sample(self):
        vertex_idx = self._sampled_vertices.pop()
        vertex = self._vertices[vertex_idx]
        return vertex, vertex_idx

    def draw_an_edge_from_sample(self):
        edge_idx = self._sampled_edges.pop()
        edge = self._graph[edge_idx]
        vertex, vertex_idx, context, context_idx, weight = edge
        return vertex, vertex_idx, context, context_idx, weight
        

    # sampler functions
    def _fast_choice(self, pool_range, amount=1):
        output_list = self._rng_generator.choice(pool_range, amount, replace=False)
        return output_list

    def draw_an_edge(self): # edge-based mode only
        edge_idx = self._fast_choice(self._edge_count)[0]
        user, item, weight = self._graph[edge_idx]
        return user, item, weight

    def draw_a_vertex(self):
        vertex_idx = self._fast_choice(self._vertex_count)[0]
        vertex = self._vertices[vertex_idx]
        return vertex, vertex_idx

    def draw_a_context(self, vertex: str):
        this_contexts = list(self._graph[vertex])
        this_context_idx_range = len(this_contexts)
        this_context_idx = self._fast_choice(this_context_idx_range)[0]
        context = this_contexts[this_context_idx]
        context_idx = self._context_idxs[context]
        return context, context_idx

    def draw_a_context_uniformly(self):
        context_idx = self._fast_choice(self._context_count)[0]
        context = self._contexts[context_idx]
        return context, context_idx

    def draw_contexts_uniformly(self, amount=5):
        context_idxs = self._fast_choice(self._context_count, amount=amount)
        contexts = [ self._contexts[context_idx] for context_idx in context_idxs ]
        return contexts, context_idxs

    def draw_all_neighbors(self, vertex: str):
        contexts = list(self._graph[vertex])
        context_idxs = [ self._context_idxs[context] for context in contexts ]
        return contexts, context_idxs

    def draw_a_walk(self, vertex: str, steps: int): #usually used for undirected graph
        walk = []
        for i in range(steps):
            vertex, vertex_idx = self.draw_a_context(vertex)
            walk.append(vertex_idx) # return index only
        return walk

    def skipgram_generator(self, vertex: str, walk_length: int, window_size: int):
        walk = self.draw_a_walk(vertex, steps=walk_length)
        reduces = self._rng_generator.integers(low=1, high=window_size+1, size=len(walk))
        for i in range(len(walk)):
            reduce = reduces[i]
            left = max(i - reduce, 0)
            right = min(i + reduce, walk_length)

            for j in range(left, right):
                if i == j:
                    continue
                yield walk[i], walk[j]


