from .util import graph_file_reader
import multiprocessing as mp

class graph:
    def __init__(self, path, delimiter='\t', bidirect=False):
        self.manager = mp.Manager()
        self.users = []
        self.items = []
        self.GRAPH = self.manager.dict()
        for user,item,weight in graph_file_reader(path=path, delimiter=delimiter):
            self.users.append(user)
            self.items.append(item)
            if user not in self.GRAPH:
                self.GRAPH[user] = {}
            #important! bypass the multiprocessing bug
            this_user = self.GRAPH[user]
            this_user[item] = weight
            self.GRAPH[user] = this_user
            ###########
            
        self.users = self.manager.list(set(self.users))
        self.items = self.manager.list(set(self.items))

        self.user_idx_range = len(self.users)
        self.item_idx_range = len(self.items)

    def update_user_idx_range(self):
        self.user_idx_range = len(self.users)

    def update_item_idx_range(self):
        self.item_idx_range = len(self.items)
    
#samplers
def 