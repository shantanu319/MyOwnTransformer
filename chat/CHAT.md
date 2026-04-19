to run the chat:

cargo run --manifest-path chat/Cargo.toml --release -- \
  --checkpoint saved/model/ckpt_final.pt \
  --data-dir /tmp/bpe_smoke \
  --no-cuda