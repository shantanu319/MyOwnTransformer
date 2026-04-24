to run the chat:

cargo run --manifest-path chat/Cargo.toml --release -- \
  --checkpoint /Users/Shantanu/Documents/GitHub/MyOwnTransformer/modal_out/saved/cosmo_5B/ckpt_step16000.pt \
  --data-dir data_cache/cosmopedia \
  --no-cuda