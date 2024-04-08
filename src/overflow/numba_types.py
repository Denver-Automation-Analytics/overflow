from numba import int64
from numba.types import UniTuple, ListType


Int64Pair = UniTuple(int64, 2)
Int64PairList = ListType(Int64Pair)
Int64PairListList = ListType(Int64PairList)
