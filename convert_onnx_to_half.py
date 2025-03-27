from onnxconverter_common import float16
import onnx

# Load float32 model
model_fp32 = onnx.load("./onnx/direct/siamfcpp_template_direct_aipp.onnx")

# Convert to float16
model_fp16 = float16.convert_float_to_float16(model_fp32)

# Save the float16 model
onnx.save(model_fp16, "onnx/half/siamfcpp_template_direct_aipp_half.onnx")


def convertONNX_search():
    # Load float32 model
    model_fp32 = onnx.load("./onnx/direct/siamfcpp_search_direct.onnx")

    # Convert to float16
    model_fp16 = float16.convert_float_to_float16(model_fp32)

    # Save the float16 model
    onnx.save(model_fp16, "onnx/half/siamfcpp_search_direct_half.onnx")


def convertONNX_template():
    # Load float32 model
    model_fp32 = onnx.load("./onnx/direct/siamfcpp_template_direct.onnx")

    # Convert to float16
    model_fp16 = float16.convert_float_to_float16(model_fp32)

    # Save the float16 model
    onnx.save(model_fp16, "onnx/half/siamfcpp_template_direct_half.onnx")

if __name__ == "__main__":
    convertONNX_search()
    convertONNX_template()
    