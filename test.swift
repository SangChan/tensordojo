#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif
print(Python.version)

import TensorFlow
var x = Tensor<Float>([[1, 2], [3, 4]])
print(x + x)