/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#ifdef USE_ESP_NN_CONV
#include "esp_nn.h"
#include "esp_nn_defs.h"
#include <esp_heap_caps.h>
#include <cstdio>
#endif

namespace tflite {
namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataConv));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data = *(static_cast<const OpDataConv*>(node->user_data));

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(
      context,
      input->type == filter->type ||
          (input->type == kTfLiteInt16 && filter->type == kTfLiteInt8),
      "Hybrid models are not supported on TFLite Micro.");

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {
      tflite::reference_ops::Conv(
          ConvParamsFloat(params, data), tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output),
          tflite::micro::GetTensorShape(nullptr), nullptr);
      break;
    }
    case kTfLiteInt16: {
      switch (bias->type) {
        case kTfLiteInt32: {
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int16_t>(input),
              tflite::micro::GetTensorShape(filter),
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<std::int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int16_t>(output));
          break;
        }
        case kTfLiteInt64: {
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int16_t>(input),
              tflite::micro::GetTensorShape(filter),
              tflite::micro::GetTensorData<int8_t>(filter),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<std::int64_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int16_t>(output));
          break;
        }
        default:
          MicroPrintf("Bias type %s (%d) not supported.",
                      TfLiteTypeGetName(bias->type), bias->type);
          return kTfLiteError;
      }
      break;
    }
    case kTfLiteInt8: {
#ifdef USE_ESP_NN_CONV
      // Fast path: ESP-NN optimized int8 conv for ESP32 family chips.
      // Falls through to the reference kernel via the #else branch when
      // USE_ESP_NN_CONV is not defined.
      static int call_count = 0;
      call_count++;
      if (call_count <= 3) {
        printf("[ESP-NN] conv2d call #%d - using fast path\n", call_count);
      }

      const auto& input_shape = tflite::micro::GetTensorShape(input);
      const auto& filter_shape = tflite::micro::GetTensorShape(filter);
      const auto& output_shape = tflite::micro::GetTensorShape(output);

      // Tensor layout is [batch, height, width, channels].
      const int input_height = input_shape.Dims(1);
      const int input_width = input_shape.Dims(2);
      const int input_depth = input_shape.Dims(3);
      const int filter_height = filter_shape.Dims(1);
      const int filter_width = filter_shape.Dims(2);
      const int output_height = output_shape.Dims(1);
      const int output_width = output_shape.Dims(2);
      const int output_depth = output_shape.Dims(3);

      data_dims_t in_dims = {input_width, input_height, input_depth, 1};
      data_dims_t fl_dims = {filter_width, filter_height, input_depth, output_depth};
      data_dims_t out_dims = {output_width, output_height, output_depth, 1};

      const ConvParams cp_tf = ConvParamsQuantized(params, data);

      conv_params_t conv_params;
      conv_params.in_offset = cp_tf.input_offset;
      conv_params.out_offset = cp_tf.output_offset;
      conv_params.stride.width = params.stride_width;
      conv_params.stride.height = params.stride_height;
      conv_params.padding.width = cp_tf.padding_values.width;
      conv_params.padding.height = cp_tf.padding_values.height;
      conv_params.dilation.width = params.dilation_width_factor;
      conv_params.dilation.height = params.dilation_height_factor;
      conv_params.activation.min = cp_tf.quantized_activation_min;
      conv_params.activation.max = cp_tf.quantized_activation_max;

      quant_data_t quant_data;
      quant_data.mult = const_cast<int32_t*>(data.per_channel_output_multiplier);
      quant_data.shift = const_cast<int32_t*>(data.per_channel_output_shift);

      // Scratch buffer is sized per-layer, reused across calls.
      // Allocated in whichever heap region has space (PSRAM is fine).
      static int8_t* scratch_buf = nullptr;
      static int scratch_size = 0;
      int needed = esp_nn_get_conv_scratch_size(
          &in_dims, &fl_dims, &out_dims, &conv_params);
      if (needed > scratch_size) {
        if (scratch_buf) heap_caps_free(scratch_buf);
        scratch_buf = (int8_t*)heap_caps_malloc(needed, MALLOC_CAP_8BIT);
        scratch_size = needed;
        if (!scratch_buf) {
          printf("[ESP-NN] FAILED to allocate %d byte scratch buffer\n", needed);
        } else if (call_count <= 3) {
          printf("[ESP-NN] allocated %d byte scratch buffer at %p\n",
                 needed, scratch_buf);
        }
      }
      esp_nn_set_conv_scratch_buf(scratch_buf);

      esp_nn_conv_s8(
          &in_dims,
          tflite::micro::GetTensorData<int8_t>(input),
          &fl_dims,
          tflite::micro::GetTensorData<int8_t>(filter),
          tflite::micro::GetOptionalTensorData<int32_t>(bias),
          &out_dims,
          tflite::micro::GetTensorData<int8_t>(output),
          &conv_params,
          &quant_data);
#else
      reference_integer_ops::ConvPerChannel(
          ConvParamsQuantized(params, data), data.per_channel_output_multiplier,
          data.per_channel_output_shift, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<int8_t>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<int32_t>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
#endif
      break;
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_CONV_2D() {
  return tflite::micro::RegisterOp(Init, ConvPrepare, Eval);
}

}  // namespace tflite
