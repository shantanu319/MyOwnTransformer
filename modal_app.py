"""Modal entrypoint: prepare TinyStories + train MyOwnTransformer on a GPU.

One-time setup:
    pip install modal
    modal setup         # opens a browser to link your account

Usage:
    # One-shot: prepare (if needed) -> train with defaults on an L4 GPU
    modal run modal_app.py

    # Override knobs (names mirror config.py / run.sh):
    modal run modal_app.py --d-model 384 --n-layers 5 --heads 6 \
        --seqlen 512 --batchsize 16 --epochs 1 --warmup-steps 300

    # Rebuild the tokenizer + .bin shards (otherwise we reuse the volume copy)
    modal run modal_app.py --force-prepare

    # Just prep, no training:
    modal run modal_app.py::prepare

    # Just train (assumes data already on the volume):
    modal run modal_app.py::train --epochs 2

    # Pull checkpoints + learning_curves back to ./modal_out/
    modal volume get myowntransformer-data /saved ./modal_out
"""
import modal


APP_NAME = "myowntransformer"
VOLUME_NAME = "myowntransformer-data"

# torch 2.11 matches what the user runs locally; torch.optim.Muon needs >= 2.9.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.0",
        "numpy",
        "datasets",
        "matplotlib",
    )
    .add_local_dir(
        ".",
        remote_path="/root/src",
        ignore=[
            "data_cache",
            "saved",
            "chat/target",
            "**/__pycache__",
            ".git",
            ".pytest_cache",
            "*.bin",
            "learning_curves.png",
            ".claude",
            "modal_out",
        ],
    )
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

VOL_MOUNT = "/vol"
DATA_DIR = f"{VOL_MOUNT}/data_cache/tinystories"
SAVE_ROOT = f"{VOL_MOUNT}/saved"


@app.function(
    volumes={VOL_MOUNT: vol},
    timeout=60 * 60 * 2,
)
def prepare(force: bool = False, vocab_size: int = 4096, bpe_train_docs: int = 10000):
    """Download TinyStories, train BPE, emit train/val/test.bin into the volume."""
    import os
    import subprocess

    if not force and os.path.exists(f"{DATA_DIR}/train.bin"):
        print(f"{DATA_DIR}/train.bin already exists — skipping (pass --force-prepare to rebuild)")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    os.chdir("/root/src")
    subprocess.run(
        [
            "python", "-u", "prepare.py",
            "--output-dir", DATA_DIR,
            "--vocab-size", str(vocab_size),
            "--bpe-train-docs", str(bpe_train_docs),
        ],
        check=True,
    )
    vol.commit()
    print(f"Data prepared and committed to volume `{VOLUME_NAME}:{DATA_DIR}`")


@app.function(
    volumes={VOL_MOUNT: vol},
    gpu="A100",
    timeout=60 * 60 * 8,
)
def train(
    d_model: int = 256,
    n_layers: int = 6,
    heads: int = 4,
    batchsize: int = 32,
    seqlen: int = 256,
    epochs: int = 1,
    lr: float = 3e-4,
    muon_lr: float = 0.02,
    warmup_steps: int = 200,
    save_every: int = 500,
    printevery: int = 50,
    dir_name: str = "modal_run",
):
    """Run train.py on a GPU, writing checkpoints + plot into the volume."""
    import os
    import shutil
    import subprocess

    if not os.path.exists(f"{DATA_DIR}/train.bin"):
        raise FileNotFoundError(
            f"no {DATA_DIR}/train.bin on the volume — run `modal run modal_app.py::prepare` first"
        )

    os.chdir("/root/src")

    # train.py hard-codes `saved/<dir_name>/` relative to CWD. Symlink that into
    # the volume so periodic checkpoints land on persistent storage directly.
    os.makedirs(SAVE_ROOT, exist_ok=True)
    if not os.path.lexists("/root/src/saved"):
        os.symlink(SAVE_ROOT, "/root/src/saved")

    env = {**os.environ, "MPLBACKEND": "Agg"}
    subprocess.run(
        [
            "python", "-u", "train.py",
            "-data_dir", DATA_DIR,
            "-dir_name", dir_name,
            "-d_model", str(d_model),
            "-n_layers", str(n_layers),
            "-heads", str(heads),
            "-batchsize", str(batchsize),
            "-seqlen", str(seqlen),
            "-epochs", str(epochs),
            "-lr", str(lr),
            "-muon_lr", str(muon_lr),
            "-warmup_steps", str(warmup_steps),
            "-save_every", str(save_every),
            "-printevery", str(printevery),
        ],
        check=True, env=env,
    )

    plot_src = "/root/src/learning_curves.png"
    if os.path.exists(plot_src):
        shutil.copy(plot_src, f"{SAVE_ROOT}/{dir_name}_learning_curves.png")
    vol.commit()
    print(f"Artifacts saved to volume `{VOLUME_NAME}:/saved/{dir_name}/`")
    print(f"Download with: modal volume get {VOLUME_NAME} /saved ./modal_out")


@app.local_entrypoint()
def main(
    force_prepare: bool = False,
    vocab_size: int = 4096,
    bpe_train_docs: int = 10000,
    d_model: int = 256,
    n_layers: int = 6,
    heads: int = 4,
    batchsize: int = 32,
    seqlen: int = 256,
    epochs: int = 1,
    lr: float = 3e-4,
    muon_lr: float = 0.02,
    warmup_steps: int = 200,
    save_every: int = 500,
    printevery: int = 50,
    dir_name: str = "modal_run",
):
    prepare.remote(force=force_prepare, vocab_size=vocab_size, bpe_train_docs=bpe_train_docs)
    train.remote(
        d_model=d_model,
        n_layers=n_layers,
        heads=heads,
        batchsize=batchsize,
        seqlen=seqlen,
        epochs=epochs,
        lr=lr,
        muon_lr=muon_lr,
        warmup_steps=warmup_steps,
        save_every=save_every,
        printevery=printevery,
        dir_name=dir_name,
    )
