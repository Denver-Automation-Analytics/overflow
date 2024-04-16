from numba import int64, float32, literal_unroll
from numba.types import Array, Record
from numba.experimental import jitclass
import numpy as np


def create_queue_class(numba_type, fields):
    spec = [
        ("_front", int64),
        ("_size", int64),
        (
            "_data",
            Array(
                numba_type,
                1,
                "C",
            ),
        ),
    ]

    @jitclass(spec)
    class Queue:
        """
        A resizable First-In-First-Out (FIFO) queue implemented with a circular buffer.
        The complexity of the push and pop operations is amortized O(1) due to the occasional resizing.
        This custom class is needed since numba does not yet support the built-in
        collections.deque class.

        Attributes
        ----------
        _front : int
            The index of the front of the queue.
        _size : int
            The current size of the queue.
        _data : np.ndarray
            The underlying data array storing the elements of the queue.
        """

        def __init__(self, data):
            """
            Initialize the queue with the given data.

            Parameters
            ----------
            data : list or np.ndarray
                The initial data to populate the queue.
            """
            self._front = 0
            self._data = np.empty(len(data), dtype=numba_type)
            for i, elem in enumerate(data):
                j = 0
                for f in literal_unroll(fields):
                    self._data[i][f] = elem[j]
                    j += 1
            self._size = self._data.shape[0]
            if self._size == 0:
                self._resize(1)

        def push(self, value):
            """
            Add a value to the end of the queue.

            Parameters
            ----------
            value : int
                The value to add to the queue.
            """
            if self._size == len(self._data):
                self._resize(2 * len(self._data))
            i = 0
            for f in literal_unroll(fields):
                self._data[(self._front + self._size) % len(self._data)][f] = value[i]
                i += 1
            self._size += 1

        def pop(self):
            """
            Remove and return the value at the front of the queue.

            Returns
            -------
            int
                The value at the front of the queue.

            Raises
            ------
            IndexError
                If the queue is empty.
            """
            if self._size == 0:
                raise IndexError("pop from an empty queue")
            value = self._data[self._front]
            self._front = (self._front + 1) % len(self._data)
            self._size -= 1
            if self._size < len(self._data) // 4:
                self._resize(len(self._data) // 2)
            return value

        def _resize(self, new_capacity):
            """
            Resize the underlying data array to the given capacity.

            Parameters
            ----------
            new_capacity : int
                The new capacity for the data array.
            """
            old_shape = self._data.shape
            new_shape = (new_capacity,) + old_shape[1:]
            new_data = np.empty(new_shape, dtype=self._data.dtype)
            old_len = min(len(self._data), new_capacity)
            new_data[:old_len] = np.concatenate(
                (self._data[self._front :], self._data[: self._front])
            )[:old_len]
            self._data = new_data
            self._front = 0

        def __len__(self):
            """
            Return the current size of the queue.

            Returns
            -------
            int
                The current size of the queue.
            """
            return self._size

    return Queue


RowColRecord = Record(
    [
        (
            "row",
            {
                "type": int64,
                "offset": 0,
                "alignment": None,
                "title": None,
            },
        ),
        (
            "col",
            {
                "type": int64,
                "offset": 8,
                "alignment": None,
                "title": None,
            },
        ),
    ],
    16,
    False,
)

RowColFields = (
    "row",
    "col",
)

RowColInt64Record = Record(
    [
        (
            "row",
            {
                "type": int64,
                "offset": 0,
                "alignment": None,
                "title": None,
            },
        ),
        (
            "col",
            {
                "type": int64,
                "offset": 8,
                "alignment": None,
                "title": None,
            },
        ),
        (
            "value",
            {
                "type": int64,
                "offset": 16,
                "alignment": None,
                "title": None,
            },
        ),
    ],
    24,
    False,
)

RowColInt64Fields = (
    "row",
    "col",
    "value",
)

RowColFloat32Record = Record(
    [
        (
            "row",
            {
                "type": int64,
                "offset": 0,
                "alignment": None,
                "title": None,
            },
        ),
        (
            "col",
            {
                "type": int64,
                "offset": 8,
                "alignment": None,
                "title": None,
            },
        ),
        (
            "value",
            {
                "type": float32,
                "offset": 16,
                "alignment": None,
                "title": None,
            },
        ),
    ],
    20,
    False,
)

RowColFloat32Fields = (
    "row",
    "col",
    "value",
)

RowColQueue = create_queue_class(RowColRecord, RowColFields)
RowColInt64Queue = create_queue_class(RowColInt64Record, RowColInt64Fields)
RowColFloat32Queue = create_queue_class(RowColFloat32Record, RowColFloat32Fields)
