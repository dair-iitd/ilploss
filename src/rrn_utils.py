import dgl
import torch
import numpy as np
from IPython.core.debugger import Pdb
from torch.utils.data.dataloader import default_collate 
import copy

constraints_graph_cache = {}
grid_cache = {}
block_shape_dict = {
                4: (2, 2),
                6: (2, 3),
                 8: (2, 4),
                 9: (3, 3),
                 10: (2, 5),
                 12: (2, 6),
                 14: (2, 7),
                 15: (3, 5),
                 16: (4, 4),
                 25: (5, 5)}


def collate_sudoku_graph(batch):
    """
    Create mini-batch data for training.
    :param data: data dict
    :return: mini-batch
    """
    graph_list = []
    board = batch[0]['num_classes'].long().item()
    block_x, block_y = block_shape_dict[board]
    sudoku_indices = np.arange(0,board*board)
    rows = sudoku_indices // board
    cols = sudoku_indices % board
    constraint_graph = create_graph(block_x,block_y)
    for this_element in batch:
        q = this_element['query/num']
        a = this_element['target/num']
        graph = copy.deepcopy(constraint_graph)
        graph.ndata['q'] = q  # q means question
        graph.ndata['a'] = a  # a means answer
        graph.ndata['row'] = torch.tensor(rows, dtype=torch.long)
        graph.ndata['col'] = torch.tensor(cols, dtype=torch.long)
        graph_list.append(graph)
    #
    mini_batch = default_collate(batch)
    mini_batch['batch_graph'] = dgl.batch(graph_list) 
    return mini_batch




def _get_constraint_graph(block_x, block_y):
   
    board_size = block_x*block_y
    num_cells = board_size*board_size

    global grid_cache    
    
    if (block_x,block_y) not in grid_cache:
        grid_cache[(block_x, block_y)] = np.array(range(num_cells)).reshape(board_size,board_size)  

    grid = grid_cache[(block_x,block_y)]
    
    edges = set()
    for i in range(num_cells):
        row, col = i // board_size, i % board_size 
        # same row and col
        row_src = row * board_size 
        col_src = col
        for _ in range(board_size):
            if row_src != i:
                edges.add((row_src, i))
            if col_src != i:
                edges.add((col_src, i))
            row_src += 1
            col_src += board_size 
        
        # same grid
        
        b_row = (i//board_size)//block_x 
        b_col = (i%board_size)//block_y

        block_values = grid[block_x*b_row:block_x*(b_row+1),block_y*b_col:block_y*(b_col+1)].flatten()       
        for n in block_values: 
            if n != i: 
                edges.add((n, i))
#         Pdb().set_trace()
    g = dgl.graph(list(edges))
    return g 


def create_graph(block_x, block_y):
    global constraints_graph_cache
    if (block_x, block_y) not in constraints_graph_cache:
        constraints_graph_cache[(block_x,block_y)] = _get_constraint_graph(block_x, block_y)
    #
    return constraints_graph_cache[(block_x,block_y)]
