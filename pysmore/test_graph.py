from libs.Graph import graph

g = graph(r'D:\repository\pySmore\pysmore\data\kktv.ui.train.txt', delimiter=' ')

user = g.draw_user()
item = g.draw_item(user[0])

print(user)
print(item)
print(g.draw_item_uniformly())