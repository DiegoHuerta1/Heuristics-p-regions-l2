import numpy as np
from multiprocessing import Pool, shared_memory


def _worker_chunk(start_idx, end_idx, shape_A, dtype_A, shm_name_A,
                  shape_B, dtype_B, shm_name_B, func):
    # Access shared memory
    shm_A = shared_memory.SharedMemory(name=shm_name_A)
    A = np.ndarray(shape_A, dtype=dtype_A, buffer=shm_A.buf)
    shm_B = shared_memory.SharedMemory(name=shm_name_B)
    B = np.ndarray(shape_B, dtype=dtype_B, buffer=shm_B.buf)
    # Process the chunk
    results = np.array([func(A[i, :], B) for i in range(start_idx, end_idx)])
    # Clean up
    shm_A.close()
    shm_B.close()
    return results


class ParallelMatrixProcessor:
    """  
    Class for parallel processing.
    Apply a function to each row of matrix A, taking matrix B as parameters.
    Both matrices are stored in shared memory.
    """
    def __init__(self, A: np.ndarray, B: np.ndarray, func,
                  pool,
                  chunk_size):
        self.func = func
        self.pool = pool
        self.chunk_size = chunk_size

        # Initialice B in shared memory
        self.dtype_B = B.dtype
        self.shape_B = B.shape
        self.shm_B = shared_memory.SharedMemory(create=True, size=B.nbytes)
        self.B_shm = np.ndarray(self.shape_B, dtype=self.dtype_B, buffer=self.shm_B.buf)
        self.B_shm[:] = B

        # Initialice A in shared memory
        self._set_shared_A(A)

    def _set_shared_A(self, A: np.ndarray):
        """
        Sets matrix A in shared memory, replacing any existing shared memory for A.
        """
        # Clean up existing shared memory for A if it exists
        if hasattr(self, 'shm_A'):
            self.shm_A.close()
            self.shm_A.unlink()
        # Set new shared memory for A
        self.dtype_A = A.dtype
        self.shape_A = A.shape
        self.shm_A = shared_memory.SharedMemory(create=True, size=A.nbytes)
        self.A_shm = np.ndarray(self.shape_A, dtype=self.dtype_A, buffer=self.shm_A.buf)
        self.A_shm[:] = A

    def execute(self) -> np.ndarray:
        """
        Executes the parallel processing,
        applying the function to each row of A with B as parameter.
        Returns:
            np.ndarray: Resulting array after processing.
        """
        tasks = []
        for start in range(0, self.shape_A[0], self.chunk_size):
            end = min(start + self.chunk_size, self.shape_A[0])
            tasks.append((start, end, self.shape_A, self.dtype_A, self.shm_A.name,
                          self.shape_B, self.dtype_B, self.shm_B.name, self.func))
        results_blocks = self.pool.starmap(_worker_chunk, tasks)
        vector_resultado = np.concatenate(results_blocks)
        return vector_resultado

    def replace_A(self, new_A: np.ndarray):
        """
        Replaces matrix A in shared memory with a new matrix.
        """
        self._set_shared_A(new_A)

    def cleanup(self):
        """
        cleans up shared memory resources.
        """
        self.shm_A.close()
        self.shm_A.unlink()
        self.shm_B.close()
        self.shm_B.unlink()
