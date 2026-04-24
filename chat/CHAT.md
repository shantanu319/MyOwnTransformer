to run the chat:

cargo run --manifest-path chat/Cargo.toml --release -- \
  --checkpoint modal_out/ckpt_step21500.pt \
  --data-dir data_cache/cosmopedia \
  --no-cuda