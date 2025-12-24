
input_length=512
output_length=128

OUTPUT_DIR=all_experiment
mkdir -p $OUTPUT_DIR


for input_length in 128 256 512 1024 2048 4096; do
  for batch_size in 1 2 4 8 16 32 64 128 256 512; do
    echo "Running batch size: $batch_size"
    echo "Running length: $input_length"

    python3 main.py --config-path hf_configs/qwen3-8B_config.json \
      --device-type H20 --world-size 1 \
      --max-prefill-tokens $input_length \
      --target-isl $input_length \
      --target-osl $output_length \
      --decode-bs $batch_size \
      --output-json $OUTPUT_DIR/input_length${input_length}_batch_size$batch_size.json
  done
done

