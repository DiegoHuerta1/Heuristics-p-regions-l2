import numpy as np
from multiprocessing import Pool, shared_memory

def _worker_chunk(start_idx, end_idx, shape_A, dtype_A, shm_name_A,
                  shape_B, dtype_B, shm_name_B, func):
    shm_A = shared_memory.SharedMemory(name=shm_name_A)
    A = np.ndarray(shape_A, dtype=dtype_A, buffer=shm_A.buf)
    
    shm_B = shared_memory.SharedMemory(name=shm_name_B)
    B = np.ndarray(shape_B, dtype=dtype_B, buffer=shm_B.buf)
    
    results = np.array([func(A[i, :], B) for i in range(start_idx, end_idx)])
    
    shm_A.close()
    shm_B.close()
    return results

class ParallelMatrixProcessor:
    def __init__(self, A: np.ndarray, B: np.ndarray, func,
                 n_workers: int = 4, chunk_size: int | None = None):
        self.func = func
        self.n_workers = n_workers

        # Inicializar B en memoria compartida
        self.dtype_B = B.dtype
        self.shape_B = B.shape
        self.shm_B = shared_memory.SharedMemory(create=True, size=B.nbytes)
        self.B_shm = np.ndarray(self.shape_B, dtype=self.dtype_B, buffer=self.shm_B.buf)
        self.B_shm[:] = B

        # Inicializar A en memoria compartida
        self._set_shared_A(A)

        # Chunk size
        self.chunk_size = chunk_size if chunk_size else max(1, A.shape[0] // n_workers)

    def _set_shared_A(self, A: np.ndarray):
        """Crea memoria compartida para A y reemplaza la anterior."""
        if hasattr(self, 'shm_A'):
            self.shm_A.close()
            self.shm_A.unlink()
        self.dtype_A = A.dtype
        self.shape_A = A.shape
        self.shm_A = shared_memory.SharedMemory(create=True, size=A.nbytes)
        self.A_shm = np.ndarray(self.shape_A, dtype=self.dtype_A, buffer=self.shm_A.buf)
        self.A_shm[:] = A

    def ejecutar(self) -> np.ndarray:
        """Ejecuta el procesamiento paralelo sobre la matriz A actual."""
        tasks = []
        for start in range(0, self.shape_A[0], self.chunk_size):
            end = min(start + self.chunk_size, self.shape_A[0])
            tasks.append((start, end, self.shape_A, self.dtype_A, self.shm_A.name,
                          self.shape_B, self.dtype_B, self.shm_B.name, self.func))

        with Pool(self.n_workers) as pool:
            results_blocks = pool.starmap(_worker_chunk, tasks)

        vector_resultado = np.concatenate(results_blocks)
        return vector_resultado

    def reemplazar_A(self, nueva_A: np.ndarray):
        """Reemplaza la matriz A actual por una nueva matriz en memoria compartida."""
        self._set_shared_A(nueva_A)

    def cleanup(self):
        """Limpia la memoria compartida."""
        self.shm_A.close()
        self.shm_A.unlink()
        self.shm_B.close()
        self.shm_B.unlink()
