/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/hlo/ir/backend_config.h"

#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

const int kNumThreads = 10000;

// This string has to be in a canonical form (without spaces and new lines)
// since the == operator does not canonicalize the raw strings before comparing
// them.
constexpr absl::string_view kRawString =
    R"({"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_gemm","triton_gemm_config":{"block_m":"256","block_n":"256","block_k":"32","split_k":"1","num_stages":"1","num_warps":"16","num_ctas":"1"}},"force_earliest_schedule":false})";

class BackendConfigWrapperTest : public testing::TestWithParam<bool> {};

TEST_P(BackendConfigWrapperTest, ConcurrentGetProtoCompareAndReset) {
  BackendConfigWrapper source(std::string{kRawString});

  absl::Notification all_threads_created;
  std::vector<std::unique_ptr<std::thread>> threads;

  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(std::make_unique<std::thread>([&] {
      all_threads_created.WaitForNotification();
      gpu::GpuBackendConfig proto;
      TF_EXPECT_OK(source.GetProto(&proto));
      EXPECT_TRUE(proto.has_fusion_backend_config());
      BackendConfigWrapper wrapped(proto);
      EXPECT_TRUE(wrapped == source);
      if (GetParam()) {
        // Depending on the parameter, the test resets the source object
        // so that the lazy initialization of the proto is also tested.
        source = BackendConfigWrapper(std::string{kRawString});
      }
    }));
  }
  all_threads_created.Notify();

  for (int i = 0; i < kNumThreads; ++i) {
    threads[i]->join();
  }
}

TEST_P(BackendConfigWrapperTest, ConcurrentGetRawStringCompareAndReset) {
  BackendConfigWrapper source_json(std::string{kRawString});
  gpu::GpuBackendConfig proto;
  TF_EXPECT_OK(source_json.GetProto(&proto));
  BackendConfigWrapper source(proto);

  absl::Notification all_threads_created;
  std::vector<std::unique_ptr<std::thread>> threads;

  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(std::make_unique<std::thread>([&] {
      all_threads_created.WaitForNotification();
      std::string raw_string = source.GetRawString();
      EXPECT_FALSE(raw_string.empty());
      BackendConfigWrapper wrapped(raw_string);
      EXPECT_TRUE(wrapped == source);

      if (GetParam()) {
        // Depending on the parameter, the test resets the source object
        // so that the lazy initialization of the raw string is also tested.
        source = BackendConfigWrapper(std::string{kRawString});
      }
    }));
  }
  all_threads_created.Notify();

  for (int i = 0; i < kNumThreads; ++i) {
    threads[i]->join();
  }
}

INSTANTIATE_TEST_SUITE_P(BackendConfigWrapperTestSuite,
                         BackendConfigWrapperTest,
                         ::testing::Values(false, true));

}  // namespace
}  // namespace xla
