@echo off
rem local build of the CUDA 12 engine for sm_61 (same flags as setup.py's build_engine, toolkit 12)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cmake -G Ninja -S "F:/Strata-0.2.12" -B "F:/Strata-0.2.12/build-cuda12" -DCMAKE_BUILD_TYPE=Release ^
 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON ^
 -DCMAKE_CUDA_ARCHITECTURES=61 ^
 -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe" ^
 -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4" ^
 -DSTRATA_GGML_DIR="F:/Strata-0.2.12/third_party/llama.cpp" ^
 -DSTRATA_EXPERIMENTAL_SM60=ON || exit /b 1
cmake --build "F:/Strata-0.2.12/build-cuda12" --target strata -j 8 || exit /b 1
echo BUILD-OK
