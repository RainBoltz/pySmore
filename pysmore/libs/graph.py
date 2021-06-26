from pysmore.libs.util import graph_reader_generator
import numpy as np
from collections import deque
from tqdm import tqdm, trange

class Graph:
    def __init__(self, path, delimiter='\t', mode='node'):
        self._rng_generator = np.random.default_rng()
        self._sampled_users = False
        self._sampled_edges = False

        self._graph = None
        self._users = []
        self._items = []
        self._user_idx_range = 0
        self._item_idx_range = 0
        self._edge_idx_range = 0

        if mode == 'node':
            self._create_graph_by_nodes(path, delimiter)
        elif mode == 'edge':
            self._create_graph_by_edges(path, delimiter)
        else:
            raise 'unknown graph type'     
    
    def _create_graph_by_nodes(self, path, delimiter):
        self._graph = {}
        for user, item, weight in tqdm(
            graph_reader_generator(path=path, delimiter=delimiter),
            desc='reading graph (node based) '):
            
            self._users.append(user)
            self._items.append(item)

            if user not in self._graph:
                self._graph[user] = {}
            self._graph[user][item] = weight

            self._edge_idx_range += 1

        self._users = list(set(self._users))
        self._items = list(set(self._items))

        self._user_idx_range = len(self._users)
        self._item_idx_range = len(self._items)
        
    
    def _create_graph_by_edges(self, path, delimiter):
        self._graph = []
        for user, item, weight in tqdm(
            graph_reader_generator(path=path, delimiter=delimiter),
            desc='reading graph (edge based) '):
            
            self._users.append(user)
            self._items.append(item)
            self._graph.append([user, -1, item, -1, weight])

        self._users = list(set(self._users))
        self._items = list(set(self._items))

        self._user_idx_range = len(self._users)
        self._item_idx_range = len(self._items)
        self._edge_idx_range = len(self._graph)

        user_mapping = { u: idx for idx, u in enumerate(self._users) }
        item_mapping = { i: idx for idx, i in enumerate(self._items) }
        for i in range(self._edge_idx_range):
            self._graph[i][1] = user_mapping[self._graph[i][0]]
            self._graph[i][3] = item_mapping[self._graph[i][2]]




    
    @property
    def user_count(self):
        return self._user_idx_range

    @property
    def item_count(self):
        return self._item_idx_range

    @property
    def edge_count(self):
        return self._edge_idx_range
    
    @property
    def users(self):
        return self._users

    @property
    def items(self):
        return self._items

    def get_weight_from_nodes(self, user, item):
        return self._graph[user][item]

    def cache_user_samples(self, amount):
        sampled_users = self._rng_generator.choice(self._user_idx_range, amount, replace=True)
        self._sampled_users = deque(sampled_users)

    def cache_edge_samples(self, amount):
        sampled_edges = self._rng_generator.choice(self._edge_idx_range, amount, replace=True)
        self._sampled_edges = deque(sampled_edges)

    def draw_a_user_from_sample(self):
        user_idx = self._sampled_users.pop()
        user = self._users[user_idx]
        return user, user_idx

    def draw_an_edge_from_sample(self):
        edge_idx = self._sampled_edges.pop()
        edge = self._graph[edge_idx]
        user, user_idx, item, item_idx, weight = edge
        return user, user_idx, item, item_idx, weight
    
    # samplers
    def _fast_choice(self, pool_range, amount=1):
        output_list = self._rng_generator.choice(pool_range, amount, replace=False)
        return output_list

    def draw_users(self, amount=1):
        user_idxs = self._fast_choice(self._user_idx_range, amount)
        users = [ self._users[user_idx] for user_idx in user_idxs ]
        return users, user_idxs
    
    def draw_a_user(self):
        user_idx = self._fast_choice(self._user_idx_range)[0]
        user = self._users[user_idx]
        return user, user_idx
    
    def draw_items(self, user: str, amount=1):
        this_items = list(self._graph[user])
        this_item_idx_range = len(this_items)
        item_idxs = self._fast_choice(this_item_idx_range, amount)
        items = [ this_items[item_idx] for item_idx in item_idxs ]
        return items, item_idxs
    
    def draw_an_item(self, user: str):
        this_items = list(self._graph[user])
        this_item_idx_range = len(this_items)
        item_idx = self._fast_choice(this_item_idx_range)[0]
        item = self._items[item_idx]
        return item, item_idx

    def draw_an_item_uniformly(self):
        item_idx = self._fast_choice(self._item_idx_range)[0]
        item = self._items[item_idx]
        return item, item_idx

    def draw_an_edge(self):
        edge_idx = self._fast_choice(self._edge_idx_range)[0]
        user, item, weight = self._graph[edge_idx]
        return user, item, weight


