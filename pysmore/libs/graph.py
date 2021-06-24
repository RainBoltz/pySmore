from .util import graph_reader_generator, fast_choice
from tqdm import tqdm

class Graph:
    def __init__(self, path, delimiter='\t', bidirect=False):
        self.users = []
        self.items = []
        self.GRAPH = {}

        for user, item, weight in tqdm(
            graph_reader_generator(path=path, delimiter=delimiter),
            desc=f'reading graph '):
            
            self.users.append(user)
            self.items.append(item)

            if user not in self.GRAPH:
                self.GRAPH[user] = {}
            self.GRAPH[user][item] = weight

            if bidirect:
                if item not in self.GRAPH:
                    self.GRAPH[item] = {}
                self.GRAPH[item][user] = weight
        
        self.users = list(set(self.users))
        self.items = list(set(self.items))

        self.user_idx_range = len(self.users)
        self.item_idx_range = len(self.items)
    
    def get_user_amount(self):
        return self.user_idx_range

    def get_item_amount(self):
        return self.item_idx_range   

    def get_weight(self, user, item):
        return self.GRAPH[user][item]

    
    # samplers
    def draw_user(self, amount=1):
        user_idxs = fast_choice(self.user_idx_range, amount)
        users = [ self.users[user_idx] for user_idx in user_idxs ]
        return users, user_idxs
    
    def draw_item(self, user: str, amount=1):
        this_items = list(self.GRAPH[user])
        this_item_idx_range = len(this_items)
        item_idxs = fast_choice(this_item_idx_range, amount)
        items = [ this_items[item_idx] for item_idx in item_idxs ]
        return items, item_idxs

    def draw_item_uniformly(self, amount=5):
        item_idxs = fast_choice(self.item_idx_range, amount)
        items = [ self.items[item_idx] for item_idx in item_idxs ]
        return items, item_idxs


