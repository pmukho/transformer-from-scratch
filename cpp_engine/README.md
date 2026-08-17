# C++ inference engine

C++ re-implementation of NumPy transformer forward pass, with no ML
dependencies. It loads the weights exported by `numpy_reference/` and must match
the NumPy reference within e-5 on both the prefill and KV-cache decode paths.

## Layout

```
cpp_engine/
├── CMakeLists.txt
├── include/engine/
│   ├── npy.hpp        # self-contained .npz/.npy reader (ZIP64-aware, no zlib/cnpy)
│   ├── tensor.hpp     # rank-N float32 tensor, 32-byte aligned buffers + views
│   ├── config.hpp     # model hyperparameters
│   ├── weights.hpp    # Weights struct (embedding, per-layer blocks, final norm)
│   ├── kv_cache.hpp   # per-layer preallocated K/V buffers
│   ├── matmul.hpp     # matmul / matmul_bt kernels
│   ├── ops.hpp        # rmsnorm, rope, softmax, swiglu, multihead kernels
│   └── model.hpp      # Model: owns weights + caches, runs prefill / decode
├── src/
│   ├── weights.cpp    # load_model(): npz -> Weights
│   ├── matmul.cpp     # kernel stubs
│   ├── ops.cpp        # kernel stubs
│   └── model.cpp      # orchestration: embed -> blocks -> norm -> lm_head
├── tools/
│   └── load_check.cpp # loads a model.npz and prints config + checksums
└── tests/
    └── parity_test.cpp # runs prefill + decode, compares against exported fixtures
```

## Build and run

Requires CMake and a C++17 compiler (g++ or clang).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Weights come from the exporter. Generate them once:

```bash
cd ../numpy_reference && uv run python scripts/export.py   # writes weights/{tiny,base}/
```

Then sanity-check the loader, and run the parity test:

```bash
./build/load_check  ../numpy_reference/weights/tiny/model.npz
./build/parity_test ../numpy_reference/weights/tiny
```

## Notes

- `np.savez` writes an uncompressed ZIP, so `npy.hpp` reads the `.npz` directly.
  It handles the ZIP64 local headers NumPy emits.
- Everything is float32, row-major. The exporter scales weights by 1/sqrt(fan_in)
  so activations stay O(1), which is what lets the scalar C++ math match NumPy to
  1e-5 despite a different summation order.
- `load_check` is a temporary tool. Once the ops exist it is replaced by a parity
  test that runs the full forward and compares against the exported fixtures.
```
