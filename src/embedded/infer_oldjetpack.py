import numpy as np
import timeit
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except:
    print("WARNING: tensorrt and pycuda are not installed!")


class ONNXClassifierWrapper_OldJetpack():
    def __init__(self, settings: dict):
        cuda.init()

        self.target_dtype = settings['fp16_mode'] and np.float16 or np.float32
        self.num_classes = settings['num_classes']

        self.load(settings['tensorrt_engine_path'])
        
        self.stream = None
        self.d_input = None
        self.d_output = None

        self.stream = None
        self.d_input = None
        self.d_output = None

    def load(self, file):
        with open(file, "rb") as f:
            self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

            self.context = self.engine.create_execution_context()
        
    def allocate_memory(self, batch):
        batch = batch.astype(self.target_dtype)

        self.output = np.empty(self.num_classes, dtype=self.target_dtype)

        self.d_input = cuda.mem_alloc(batch.nbytes)
        self.d_output = cuda.mem_alloc(self.output.nbytes)

        binding_names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
        
        bindings = [0] * self.engine.num_bindings
        for i, name in enumerate(binding_names):
            binding_idx = self.engine.get_binding_index(name)
            
            if self.engine.binding_is_input(binding_idx):
                try:
                    self.context.set_binding_shape(binding_idx, batch.shape)
                except Exception as e:
                    print(f"Error setting binding shape: {e}")
                bindings[binding_idx] = int(self.d_input)
            else:
                bindings[binding_idx] = int(self.d_output)

        self.stream = cuda.Stream()
        self.bindings = bindings

    def predict(self, batch):
        if self.stream is None:
            self.allocate_memory(batch)

        try:
            cuda.memcpy_htod_async(self.d_input, batch, self.stream)
            
            start_time = timeit.default_timer()
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            elapsed_time = timeit.default_timer() - start_time

            cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
            self.stream.synchronize()
        except Exception as e:
            print(f"Prediction error: {e}")
            raise

        return self.output, elapsed_time
