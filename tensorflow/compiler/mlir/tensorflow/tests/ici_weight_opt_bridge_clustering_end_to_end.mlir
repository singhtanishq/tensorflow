// RUN: tf-opt %s -tf-replicated-clustering-bridge-v2 | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK: %[[Read0:.*]] = "tf.ReadVariableOp"(%arg0)
// CHECK: %[[Read1:.*]] = "tf.ReadVariableOp"(%arg1)
// CHECK-COUNT-6: tf.Fill
// CHECK: tf_device.replicate
// CHECK: [%[[Read0]], %[[Fill1:.*]], %[[Fill2:.*]], %[[Fill3:.*]]] as %[[Arg1:.*]]: tensor<25x5xf32>, [%[[Read1]], %[[Fill4:.*]], %[[Fill5:.*]], %[[Fill6:.*]]] as %[[Arg2:.*]]: tensor<25x5xf32>
// CHECK: %[[Launch1:.*]] = "tf_device.launch"
// CHECK: %[[Launch2:.*]] = "tf_device.launch"
// CHECK: "tf_device.cluster_func"(%[[Launch1]], %[[Launch2]])
// CHECK-NOT: tf.ReadVariableOp
// CHECK-LABEL: func.func private @_func
// CHECK-COUNT-2: tf.XlaAllReduce

module attributes {tf.devices = {"/job:tpu_host_worker/replica:0/task:0/device:CPU:0", "/job:tpu_host_worker/replica:0/task:0/device:TPU:0", "/job:tpu_host_worker/replica:0/task:0/device:TPU:1", "/job:tpu_host_worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:tpu_host_worker/replica:0/task:1/device:CPU:0", "/job:tpu_host_worker/replica:0/task:1/device:TPU:0", "/job:tpu_host_worker/replica:0/task:1/device:TPU:1", "/job:tpu_host_worker/replica:0/task:1/device:TPU_SYSTEM:0"}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1850 : i32}} {
  func.func @main(%arg0: tensor<*x!tf_type.resource<tensor<25x5xf32>>> {tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"},
                  %arg1: tensor<*x!tf_type.resource<tensor<25x5xf32>>> {tf.device = "/job:tpu_host_worker/replica:0/task:1/device:CPU:0"})
                  -> (tensor<*xf32>, tensor<*xf32>) attributes {allow_soft_placement = false, tf.entry_function = {control_outputs = "", inputs = "steps,unknown,unknown_0,unknown_1,unknown_2,unknown_3,unknown_4,unknown_5,unknown_6,unknown_7,unknown_8,unknown_9,unknown_10,unknown_11,unknown_12,unknown_13", outputs = "statefulpartitionedcall_RetVal"}} {
    %0:2 = tf_executor.graph {
      %outputs_0:2, %control_1 = tf_executor.island wraps "tf.StatefulPartitionedCall"(%arg0, %arg1) <{config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\00\82\01\05h\01\88\01\01", executor_type = "", f = @_func}> {_collective_manager_ids = [], _read_only_resource_inputs = [8, 9, 10, 11, 12, 13], device = ""} : (tensor<*x!tf_type.resource<tensor<25x5xf32>>>, tensor<*x!tf_type.resource<tensor<25x5xf32>>>) -> (tensor<*xf32>,tensor<*xf32>)
      tf_executor.fetch %outputs_0#0, %outputs_0#1 : tensor<*xf32>, tensor<*xf32>
    }
    return %0#0, %0#1: tensor<*xf32>, tensor<*xf32>
  }

  func.func private @_func(%arg0: tensor<!tf_type.resource>, %arg1: tensor<!tf_type.resource>) -> (tensor<*xf32>, tensor<*xf32>) attributes {tf._construction_context = "kEagerRuntime", tf._disable_acd = true, tf.signature.is_stateful} {
    %0:2 = tf_executor.graph {
      %control = tf_executor.island wraps "tf.NoOp"() {_pivot_for_cluster = "cluster__train_helper", device = ""} : () -> ()
      %control_0 = tf_executor.island(%control) wraps "tf.NoOp"() {_has_manual_control_dependencies = true, _tpu_replicate = "cluster__train_helper", device = ""} : () -> ()
      %control_1 = tf_executor.island(%control) wraps "tf.TPUReplicateMetadata"() <{allow_soft_placement = false, computation_shape = [], device_assignment = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], host_compute_core = [], num_cores_per_replica = 1 : i64, num_replicas = 4 : i64, padding_map = [], step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\02\02\01\01\10\02\18\02\22\10\00\00\00\00\00\01\00\00\01\00\00\00\01\01\00\00*\02\08\01", tpu_compile_options_proto = "", use_spmd_for_xla_partitioning = true, use_tpu = true}> {_has_manual_control_dependencies = true, _tpu_replicate = "cluster__train_helper", device = ""} : () -> ()
      %outputs, %control_2 = tf_executor.island(%control_1) wraps "tf.TPUCompilationResult"() {_tpu_compilation_status = "cluster__train_helper", device = ""} : () -> tensor<!tf_type.string>
      %outputs_3, %control_4 = tf_executor.island wraps "tf.ReadVariableOp"(%arg0) {_tpu_replicate = "cluster__train_helper", device = ""} : (tensor<!tf_type.resource>) -> tensor<*xf32>
      %outputs_5, %control_6 = tf_executor.island wraps "tf.Identity"(%outputs_3) {_tpu_replicate = "cluster__train_helper", device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_7, %control_8 = tf_executor.island wraps "tf.Const"() <{value = dense<-1> : tensor<1xi32>}> {device = ""} : () -> tensor<1xi32>
      %outputs_9, %control_10 = tf_executor.island wraps "tf.Const"() <{value = dense<[[0, 3]]> : tensor<1x2xi32>}> {device = ""} : () -> tensor<1x2xi32>
      %outputs_11, %control_12 = tf_executor.island wraps "tf.Reshape"(%outputs_5, %outputs_7) {_tpu_replicate = "cluster__train_helper", device = ""} : (tensor<*xf32>, tensor<1xi32>) -> tensor<*xf32>
      %outputs_13, %control_14 = tf_executor.island wraps "tf.Pad"(%outputs_11, %outputs_9) {_tpu_replicate = "cluster__train_helper", device = ""} : (tensor<*xf32>, tensor<1x2xi32>) -> tensor<*xf32>
      %outputs_15, %control_16 = tf_executor.island wraps "tf.Identity"(%outputs_13) {_tpu_output_identity = true, _tpu_replicate = "cluster__train_helper", device = "/device:TPU_REPLICATED_CORE:0"} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_17, %control_18 = tf_executor.island wraps "tf.ReadVariableOp"(%arg1) {_tpu_replicate = "cluster__train_helper", device = ""} : (tensor<!tf_type.resource>) -> tensor<*xf32>
      %outputs_19, %control_20 = tf_executor.island wraps "tf.Identity"(%outputs_17) {_tpu_replicate = "cluster__train_helper", device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_21, %control_22 = tf_executor.island wraps "tf.Const"() <{value = dense<-1> : tensor<1xi32>}> {device = ""} : () -> tensor<1xi32>
      %outputs_23, %control_24 = tf_executor.island wraps "tf.Const"() <{value = dense<[[0, 3]]> : tensor<1x2xi32>}> {device = ""} : () -> tensor<1x2xi32>
      %outputs_25, %control_26 = tf_executor.island wraps "tf.Reshape"(%outputs_19, %outputs_21) {_tpu_replicate = "cluster__train_helper", device = ""} : (tensor<*xf32>, tensor<1xi32>) -> tensor<*xf32>
      %outputs_27, %control_28 = tf_executor.island wraps "tf.Pad"(%outputs_25, %outputs_23) {_tpu_replicate = "cluster__train_helper", device = ""} : (tensor<*xf32>, tensor<1x2xi32>) -> tensor<*xf32>
      %outputs_29, %control_30 = tf_executor.island wraps "tf.Identity"(%outputs_27) {_tpu_output_identity = true, _tpu_replicate = "cluster__train_helper", device = "/device:TPU_REPLICATED_CORE:0"} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_31:4, %control_32 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%outputs_15) {device = ""} : (tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
      %outputs_33:4, %control_34 = tf_executor.island wraps "tf.TPUReplicatedOutput"(%outputs_29) {device = ""} : (tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
      %outputs_35, %control_36 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs_31#0) {_has_manual_control_dependencies = true, device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      %control_37 = tf_executor.island(%control_36) wraps "tf.NoOp"() {_has_manual_control_dependencies = true, device = ""} : () -> ()
      %outputs_38, %control_39 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs_31#1) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_40, %control_41 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs_31#2) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_42, %control_43 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs_31#3) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_44, %control_45 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs_33#0) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_46, %control_47 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs_33#1) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_48, %control_49 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs_33#2) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      %outputs_50, %control_51 = tf_executor.island(%control_0) wraps "tf.Identity"(%outputs_33#3) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      tf_executor.fetch %outputs_38, %outputs_50 : tensor<*xf32>, tensor<*xf32>
    }
    return %0#0, %0#1 : tensor<*xf32>, tensor<*xf32>
  }
}