# GPT2
A basic implementation of the GPT2 Model (adapted from the full encoder-decoder transformer). This is done in a single python file, which also has a training harness so that you can build it on any datasets you like. I used wikitext-2.


to run the pipeline (train, test, and validate):
    ./run.sh                          # defaults: ~8M params, 1 epoch, full TinyStories
    EPOCHS=3 D_MODEL=128 ./run.sh     # override any knob via env
    FORCE_PREPARE=1 ./run.sh          # rebuild BPE + .bin shards

Total Run:
D_MODEL=384 N_LAYERS=5 HEADS=6 \
SEQLEN=512 BATCHSIZE=16 \
EPOCHS=1 WARMUP_STEPS=300 \
./run.sh