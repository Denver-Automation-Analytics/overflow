import numpy as np
from overflow.flow_accumulation import generate_flow_accumulation

global_graph = generate_flow_accumulation(
    "/workspaces/overflow/data/HDEM_mnbr_flow_dir.tif",
    "/workspaces/overflow/data/HDEM_mnbr_flow_acc.tif",
    chunk_size=500,
)


print(global_graph)
