CPU_GPU_Mergesort
Minimum spec: SM 3.5

A project to do mergesort within CPU and GPU, inplemention of CUDA 11.1.

Known problems:
In WDDM mode, most of the GPU resource has been used to service for display unit, and lack of efficacy to do other calculation, the result of the execution time may be unreal and not helpful, to increse the efficacy, the GPU driver have to be switched to TCC mode.
