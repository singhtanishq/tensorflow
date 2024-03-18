licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])

cc_import(
    name = "cublas",
    hdrs = [":headers"],
    shared_library = "lib/libcublas.so.%{version}",
    visibility = ["//visibility:public"],
)

cc_import(
    name = "cublasLt",
    hdrs = [":headers"],
    shared_library = "lib/libcublasLt.so.%{version}",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    hdrs = [
      "include/cublas.h", 
      "include/cublas_v2.h", 
      "include/cublas_api.h", 
      "include/cublasLt.h"
    ],
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
