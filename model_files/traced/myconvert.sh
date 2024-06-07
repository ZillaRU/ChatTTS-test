source /workspace/tpu-mlir/envsetup.sh
mkdir tmp

pushd tmp

mkdir mlir_files
model_transform.py --model_name chattts_decoder \
--model_def ../dec_1-768-1024.onnx \
--input_shapes [[1,768,1024]] \
--mlir mlir_files/chattts_decoder.mlir

model_transform.py --model_name chattts_vocos \
--model_def ../vocos_1-100-2048.onnx \
--input_shapes [[1,100,2048]] \
--mlir mlir_files/chattts_vocos.mlir

model_deploy.py --mlir mlir_files/chattts_decoder.mlir \
--model ../dec_1-768-1024.bmodel \
--quantize BF16 \
--chip bm1684x

model_deploy.py --mlir mlir_files/chattts_vocos.mlir \
--model ../vocos_1-100-2048.bmodel \
--quantize BF16 \
--chip bm1684x

popd