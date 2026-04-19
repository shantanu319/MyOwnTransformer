to run the chat:

cargo run --manifest-path chat/Cargo.toml --release -- \
  --checkpoint saved/model/ckpt_step6000.pt \
  --data-dir data_cache/tinystories \
  --no-cuda