from numba import int64
from numba.experimental import jitclass
import numpy as np

spec = [
    ("_front", int64),
    ("_size", int64),
    ("_data", int64[:, :]),
]


@jitclass(spec)
class ResizableFIFOQueue:
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
        self._data = np.array(data, dtype=np.int64)
        self._size = self._data.shape[0]

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
        self._data[(self._front + self._size) % len(self._data)] = value
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
