"""Modal entrypoint: prepare a corpus + train MyOwnTransformer on a GPU.

One-time setup:
    pip install modal
    modal setup         # opens a browser to link your account

Usage:
    # One-shot: prepare (if needed) -> train with ~90M defaults on H100.
    modal run modal_app.py

    # Override knobs:
    modal run modal_app.py --d-model 640 --n-layers 14 --heads 10 \
        --seqlen 1024 --batchsize 128 --epochs 1 --warmup-steps 1000

    # Rebuild the tokenizer + .bin shards (otherwise we reuse the volume copy)
    modal run modal_app.py --force-prepare

    # Just prep, no training:
    modal run modal_app.py::prepare

    # Just train (assumes data already on the volume):
    modal run modal_app.py::train --dir-name my_run

    # Detach (don't block the shell; stream logs with `modal app logs`):
    modal run --detached modal_app.py

    # Pull checkpoints + learning_curves back to ./modal_out/
    modal volume get myowntransformer-data /saved ./modal_out
"""
import modal


APP_NAME = "myowntransformer"
VOLUME_NAME = "myowntransformer-data"
VOL_MOUNT = "/vol"
SAVE_ROOT = f"{VOL_MOUNT}/saved"


DATA_DIR = f"{VOL_MOUNT}/data_cache/cosmopedia"

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


@app.function(
    volumes={VOL_MOUNT: vol},
    cpu=4,
    timeout=60 * 60 * 6,
)
def prepare(
    force: bool = False,
    vocab_size: int = 32000,
    bpe_train_docs: int = 10000,
    max_train_docs: int = 0,
    holdout_period: int = 500,
):
    """Download cosmopedia, train BPE, emit train/val/test.bin into the volume.

    max_train_docs=0 means no cap (full stream)."""
    import os
    import subprocess

    if not force and os.path.exists(f"{DATA_DIR}/train.bin"):
        print(f"{DATA_DIR}/train.bin already exists — skipping (pass --force-prepare to rebuild)")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    os.chdir("/root/src")
    cmd = [
        "python", "-u", "prepare.py",
        "--output-dir", DATA_DIR,
        "--vocab-size", str(vocab_size),
        "--bpe-train-docs", str(bpe_train_docs),
        "--holdout-period", str(holdout_period),
    ]
    if max_train_docs > 0:
        cmd += ["--max-train-docs", str(max_train_docs)]
    subprocess.run(cmd, check=True)
    vol.commit()
    print(f"Data prepared and committed to volume `{VOLUME_NAME}:{DATA_DIR}`")


@app.function(
    volumes={VOL_MOUNT: vol},
    gpu="H100",
    timeout=60 * 60 * 24,
)
def train(
    d_model: int = 640,
    n_layers: int = 14,
    heads: int = 10,
    batchsize: int = 128,
    seqlen: int = 1024,
    epochs: int = 1,
    lr: float = 3e-4,
    muon_lr: float = 0.02,
    warmup_steps: int = 1000,
    save_every: int = 2000,
    printevery: int = 50,
    dir_name: str = "modal_run",
):
    """Run train.py on a GPU, writing checkpoints + plot into the volume."""
    import os
    import shutil
    import subprocess

    if not os.path.exists(f"{DATA_DIR}/train.bin"):
        raise FileNotFoundError(
            f"no {DATA_DIR}/train.bin on the volume — "
            f"run `modal run modal_app.py::prepare` first"
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
    vocab_size: int = 32000,
    bpe_train_docs: int = 10000,
    max_train_docs: int = 0,
    holdout_period: int = 500,
    d_model: int = 640,
    n_layers: int = 14,
    heads: int = 10,
    batchsize: int = 128,
    seqlen: int = 1024,
    epochs: int = 1,
    lr: float = 3e-4,
    muon_lr: float = 0.02,
    warmup_steps: int = 1000,
    save_every: int = 2000,
    printevery: int = 50,
    dir_name: str = "modal_run",
):
    prepare.remote(
        force=force_prepare,
        vocab_size=vocab_size,
        bpe_train_docs=bpe_train_docs,
        max_train_docs=max_train_docs,
        holdout_period=holdout_period,
    )
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
