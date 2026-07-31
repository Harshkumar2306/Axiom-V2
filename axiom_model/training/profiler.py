import time
import torch

class Profiler:
    def __init__(self):
        self.step_start_time = None
        self.batch_tokens = 0
        
    def start_step(self, batch_size, seq_len):
        self.step_start_time = time.time()
        self.batch_tokens = batch_size * seq_len
        
    def end_step(self):
        if self.step_start_time is None:
            # start_step was never called (e.g. resumed mid-accumulation window)
            return {"time_sec": 0.0, "tokens_per_sec": 0.0, "examples_per_sec": 0.0}
        elapsed = time.time() - self.step_start_time
        tokens_per_sec = self.batch_tokens / elapsed if elapsed > 0 else 0
        examples_per_sec = (self.batch_tokens / 4096) / elapsed if elapsed > 0 else 0
        return {
            "time_sec": elapsed,
            "tokens_per_sec": tokens_per_sec,
            "examples_per_sec": examples_per_sec
        }

    def get_gpu_memory(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            return {"allocated_gb": allocated, "reserved_gb": reserved}
        return {"allocated_gb": 0, "reserved_gb": 0}
