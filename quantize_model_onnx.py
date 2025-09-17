from onnxruntime.quantization import quantize_dynamic, QuantType, create_calibrator
import onnx

# Load the original ONNX model file path
onnx_model_path = 'best_80map.onnx'
quantized_model_path = 'quantized_model1.onnx'

# Specify the nodes/ops to be quantized
nodes_to_quantize = ['Conv', 'MatMul']  # Common layer types to quantize
nodes_to_exclude = []  # Layers you want to skip

# Create quantization configuration
config = {
    'operators_to_quantize': nodes_to_quantize,
    'nodes_to_exclude': nodes_to_exclude,
    'per_channel': False,
    'reduce_range': False,
    'weight_type': QuantType.QUInt8
}

# Quantize the model with specific configuration
quantize_dynamic(
    model_input=onnx_model_path,
    model_output=quantized_model_path,
    per_channel=config['per_channel'],
    reduce_range=config['reduce_range'],
    weight_type=config['weight_type'],
    op_types_to_quantize=config['operators_to_quantize'],  # Changed parameter name
    nodes_to_exclude=config['nodes_to_exclude']
)

# Verify the quantization
model = onnx.load(quantized_model_path)
print(f"Quantization completed. Model saved to: {quantized_model_path}")
print(f"Quantized layers: {nodes_to_quantize}")
print(f"Excluded layers: {nodes_to_exclude}")