from multiprocessing import Pool

# Readers
def graph_file_reader(path, delimiter='\t'):
    with open(path) as f:
        for line in f:
            user, item, weight = line.rstrip().split(delimiter)
            yield user, item, weight

# 