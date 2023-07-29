from pathlib import Path

from animatediff import get_dir

EMBED_EXTS = [".pt", ".pth", ".bin", ".safetensors"]

embedding_dir = get_dir("data").joinpath("embeddings")


def scan_text_embeddings() -> list[Path]:
    return [x for x in embedding_dir.iterdir() if x.suffix.lower() in EMBED_EXTS]


def get_text_embeddings() -> list[tuple[str, Path]]:
    return [(x.stem, x) for x in scan_text_embeddings()]
