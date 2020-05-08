#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif
print(Python.version)