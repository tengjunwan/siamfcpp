from pathlib import Path


import onnx
from onnxsim import simplify
import onnxoptimizer


# optimize
def opt_onnx(onnx_load_path, onnx_save_path):
    # Load ONNX model
    model = onnx.load(onnx_load_path)

    # Step 1: Simplify (fix shapes, remove redundant ops)
    model, check = simplify(model)

    # Step 2: Optimize (fuse layers, runtime optimizations)
    # passes = ["fuse_bn_into_conv", "eliminate_identity", "fuse_matmul_add"]
    passes = [
        "eliminate_identity",   # Remove redundant identity ops
        "eliminate_deadend",    # Remove unused outputs
        "fuse_bn_into_conv",    # Fuse BatchNorm into Conv
        "eliminate_nop_dropout", # Remove unnecessary Dropout layers
        "eliminate_unused_initializer", # Remove unused initializers
        ]
    model = onnxoptimizer.optimize(model, passes)

    # Save final model
    onnx.save(model, onnx_save_path)
    print("✅ Model simplified and optimized!")


if __name__ == "__main__":
    # Load the ONNX model
    # query_onnx_path = "./STMTrack_FeatureExtractionQuery.onnx"
    # memory_onnx_path = "./STMTrack_FeatureExtractionMemory.onnx"l
    # head_onnx_path = "./STMTrack_ReadMemoryAndHead.onnx"


    opt_flag = True
    if opt_flag:
        template_onnx_path = "./onnx/direct/siamfcpp_template_direct.onnx"
        opt_template_onnx_path = "./onnx/opt/siamfcpp_template_opt.onnx"
        opt_onnx(template_onnx_path, opt_template_onnx_path)

        search_onnx_path = "./onnx/direct/siamfcpp_search_direct.onnx"
        opt_search_onnx_path = "./onnx/opt/siamfcpp_search_opt.onnx"
        opt_onnx(search_onnx_path, opt_search_onnx_path)

