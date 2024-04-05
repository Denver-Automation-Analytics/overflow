from numba import int64, int32, float32
from numba.typed import List, Dict
from numba.types import UniTuple, Tuple, Array, ListType, DictType


Int64Pair = UniTuple(int64, 2)
Int64PairList = ListType(Int64Pair)
Int64PairListList = ListType(Int64PairList)
