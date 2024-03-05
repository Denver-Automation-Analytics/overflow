import pytest
import numpy as np
from overflow.breach_pits import breach_single_cell_pits



def test_breach_cingle_cell_pits():
    
    chunk = np.array(
        [
            [100, 99, 90, 97, 96],
            [99, 98, 97, 96, 95],
            [98, 97, 80, 95, 94],
            [97, 96, 95, 94, 93],
            [96, 90, 85, 40, 92],
        ]
    )
    breach=breach_single_cell_pits(chunk)
    assert breach[3,3]==60