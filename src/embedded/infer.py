import numpy as np
import timeit
try:
    import tensorrt as trt
    import pycuda.driver as cuda
except:
    print("WARNING: tensorrt and pycuda are not installed!")

class ONNXClassifierWrapper():
    def __init__(self, settings: dict):
        cuda.init()
        
        self.target_dtype = settings['fp16_mode'] and np.float16 or np.float32
        self.num_classes = settings['num_classes']

        self.load(settings['tensorrt_engine_path'])
        
        self.stream = None
        self.d_input = None
        self.d_output = None

    def load(self, engine_path: str):
        engine_file = open(engine_path, "rb")
        content = engine_file.read()

        if len(content) == 0:
            engine_file.seek(0)
            content = engine_file.read()
        assert len(content) > 0, "TensorRT engine file is empty!"

        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
        self.engine = self.runtime.deserialize_cuda_engine(content)
        self.context = self.engine.create_execution_context()
        engine_file.close()
        
    def allocate_memory(self, batch):
        self.output = np.empty(self.num_classes, dtype=self.target_dtype) 
        
        self.d_input = cuda.mem_alloc(1 * batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)
        
        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        assert(len(tensor_names) == 2)

        self.context.set_tensor_address(tensor_names[0], int(self.d_input))
        self.context.set_tensor_address(tensor_names[1], int(self.d_output))

        self.stream = cuda.Stream()
    
    def predict(self, batch):
        if self.stream is None:
            self.allocate_memory(batch)

        cuda.memcpy_htod_async(self.d_input, batch, self.stream)

        start_time = timeit.default_timer()
        self.context.execute_async_v3(self.stream.handle)
        elapsed_time = timeit.default_timer() - start_time

        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.output, elapsed_time
