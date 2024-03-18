# Copyright 2024 The TensorFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for defining CUDA and cuDNN JSON files with distributives versions."""

load("//third_party:repo.bzl", "tf_mirror_urls")

_DEFAULT_CUDA_VERSION = "12.3.2"
_DEFAULT_CUDNN_VERSION = "8.9.7.29"

def _get_env_var(ctx, name):
    if name in ctx.os.environ:
        return ctx.os.environ[name]
    else:
        return None

def _cuda_redist_json_impl(repository_ctx):
    cuda_version = _get_env_var(repository_ctx, "TF_CUDA_VERSION")
    cudnn_version = _get_env_var(repository_ctx, "TF_CUDNN_VERSION")
    supported_cuda_versions = repository_ctx.attr.cuda_json_dict.keys()
    if cuda_version and (cuda_version not in supported_cuda_versions):
        if cuda_version in ["12", "12.3"]:
            cuda_version = _DEFAULT_CUDA_VERSION
        else:
            fail(
                ("The supported CUDA versions are {supported_versions}." +
                 " Please provide a supported version in TF_CUDA_VERSION" +
                 " environment variable or add JSON URL for" +
                 " CUDA version={version}.")
                    .format(
                    supported_versions = supported_cuda_versions,
                    version = cuda_version,
                ),
            )
    supported_cudnn_versions = repository_ctx.attr.cudnn_json_dict.keys()
    if cudnn_version and (cudnn_version not in supported_cudnn_versions):
        if cudnn_version in ["8", "8.9"]:
            cudnn_version = _DEFAULT_CUDNN_VERSION
        else:
            fail(
                ("The supported CUDNN versions are {supported_versions}." +
                 " Please provide a supported version in TF_CUDNN_VERSION" +
                 " environment variable or add JSON URL for" +
                 " CUDNN version={version}.")
                    .format(
                    supported_versions = supported_cudnn_versions,
                    version = cudnn_version,
                ),
            )
    cuda_distributives = "{}"
    cudnn_distributives = "{}"
    if cuda_version:
        (url, sha256) = repository_ctx.attr.cuda_json_dict[cuda_version]
        json_file_name = "redistrib_cuda_%s.json" % cuda_version
        repository_ctx.download(
            url = tf_mirror_urls(url),
            sha256 = sha256,
            output = json_file_name,
        )
        cuda_distributives = repository_ctx.read(repository_ctx.path(json_file_name))
    if cudnn_version:
        (url, sha256) = repository_ctx.attr.cudnn_json_dict[cudnn_version]
        json_file_name = "redistrib_cudnn_%s.json" % cudnn_version
        repository_ctx.download(
            url = tf_mirror_urls(url),
            sha256 = sha256,
            output = json_file_name,
        )
        cudnn_distributives = repository_ctx.read(repository_ctx.path(json_file_name))

    repository_ctx.file(
        "build_defs.bzl",
        """def get_cuda_distributives():
               return {cuda_distributives}

def get_cudnn_distributives():
    return {cudnn_distributives}
""".format(cuda_distributives = cuda_distributives, cudnn_distributives = cudnn_distributives),
    )
    repository_ctx.file(
        "BUILD",
        "",
    )

_cuda_redist_json = repository_rule(
    implementation = _cuda_redist_json_impl,
    attrs = {
        "cuda_json_dict": attr.string_list_dict(mandatory = True),
        "cudnn_json_dict": attr.string_list_dict(mandatory = True),
    },
    environ = ["TF_CUDA_VERSION", "TF_CUDNN_VERSION"],
)

def cuda_redist_json(name, cuda_json_dict, cudnn_json_dict):
    _cuda_redist_json(
        name = name,
        cuda_json_dict = cuda_json_dict,
        cudnn_json_dict = cudnn_json_dict,
    )
