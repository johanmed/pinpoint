"""Module with UCE-based CellxGene tools for Pinpoint

Unlike scVI, UCE has no Census-hosted vector index, so a reference of Census
cells is embedded once and cached to disk on first use (see build_reference).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import anndata as ad
import cellxgene_census
import dspy
import numpy as np
import pandas as pd
import scipy.sparse as sp

EMBEDDING_NAME = "X_uce"
CENSUS_VERSION = "2025-01-30"
K = 30
N_REFERENCE_CELLS = 3000
BATCH_SIZE = 16

CACHE_DIR = Path.home() / ".cache" / "pinpoint-uce"
UCE_REPO = CACHE_DIR / "UCE"
UCE_URL = "https://github.com/snap-stanford/UCE"
ALLOW_DOWNLOAD = True  # set False to forbid cloning/fetching anything

SPECIES = {
    "homo_sapiens": "human",
    "mus_musculus": "mouse",
    "macaca_mulatta": "macaca_mulatta",
    "macaca_fascicularis": "macaca_fascicularis",
    "sus_scrofa": "pig",
    "microcebus_murinus": "mouse_lemur",
    "danio_rerio": "zebrafish",
    "xenopus_tropicalis": "frog",
}

ASSETS = {
    "species_chrom.csv": "https://figshare.com/ndownloader/files/42706558",
    "species_offsets.pkl": "https://figshare.com/ndownloader/files/42706555",
    "all_tokens.torch": "https://figshare.com/ndownloader/files/42706585",
    "4layer_model.torch": "https://figshare.com/ndownloader/files/42706576",
}
PROTEIN_EMBEDDINGS_URL = "https://figshare.com/ndownloader/files/42715213"

# UCE's 4-layer defaults. filter must stay False: its filter_genes(min_cells=10)
# would delete every gene of a single-cell pseudo-cell, and the upstream CLI
# cannot switch it off (argparse type=bool turns "--filter False" into True).
UCE_KWARGS = {
    "filter": False,
    "skip": False,
    "pad_length": 1536,
    "pad_token_idx": 0,
    "chrom_token_left_idx": 1,
    "chrom_token_right_idx": 2,
    "cls_token_idx": 3,
    "CHROM_TOKEN_OFFSET": 143574,
    "sample_size": 1024,
    "CXG": True,
    "nlayers": 4,
    "output_dim": 1280,
    "d_hid": 5120,
    "token_dim": 5120,
    "multi_gpu": False,
}


def resolve_species(organism: str) -> str:
    """Map a Census organism name to a UCE species name."""
    key = str(organism).strip().replace(" ", "_").lower()
    if key in SPECIES:
        return SPECIES[key]
    for census_name, uce_name in SPECIES.items():
        if key == uce_name:
            return uce_name
    raise ValueError(
        f"UCE has no gene embeddings for {organism!r}. Supported: {sorted(SPECIES)}. "
        "Rat needs the upstream 'Create New Species Files' pipeline; use cellxgene.py "
        "(Census scVI) for rat data instead."
    )


def ensure_assets(allow_download: bool | None = None) -> Path:
    """Check the UCE repo and model files, fetching whatever is missing.

    Nothing needs to be configured by hand: the source is cloned from GitHub and
    the model bundle (~6.4 GB) is pulled from figshare into CACHE_DIR. Pass
    ``allow_download=False`` (or set the module constant) to keep it offline.
    """
    if allow_download is None:
        allow_download = ALLOW_DOWNLOAD
    if not (UCE_REPO / "evaluate.py").exists():
        if not allow_download:
            raise RuntimeError(
                f"UCE source is missing from {UCE_REPO} and ALLOW_DOWNLOAD is off; "
                f"git clone {UCE_URL} there"
            )
        print(f"Cloning {UCE_URL} into {UCE_REPO}")
        UCE_REPO.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", UCE_URL, str(UCE_REPO)], check=True
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"Could not clone {UCE_URL} into {UCE_REPO}: {exc}"
            ) from exc

    model_dir = CACHE_DIR / "model_files"
    model_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in ASSETS if not (model_dir / name).exists()] + (
        [] if (model_dir / "protein_embeddings").exists() else ["protein_embeddings/"]
    )
    if missing and not allow_download:
        raise RuntimeError(
            f"UCE assets are missing from {model_dir}: {missing} and ALLOW_DOWNLOAD "
            "is off; set ALLOW_DOWNLOAD = True to fetch ~6.4 GB from figshare"
        )

    for name in missing:
        if name == "protein_embeddings/":
            archive = model_dir / "protein_embeddings.tar.gz"
            if not archive.exists():
                print(f"Downloading {archive.name} from figshare")
                urllib.request.urlretrieve(PROTEIN_EMBEDDINGS_URL, archive)
            print(f"Extracting {archive.name}")
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(model_dir)
            archive.unlink()
        else:
            print(f"Downloading {name} from figshare")
            urllib.request.urlretrieve(ASSETS[name], model_dir / name)

    return model_dir


def embed_anndata(
    adata: ad.AnnData,
    species: str,
    workspace: Path,
    batch_size: int = BATCH_SIZE,
) -> ad.AnnData:
    """Embed cells with UCE; returns the processed AnnData with obsm["X_uce"]."""
    import torch
    from accelerate import Accelerator

    model_dir = ensure_assets()
    workspace.mkdir(parents=True, exist_ok=True)
    adata_path = workspace / "input.h5ad"
    adata.write(adata_path)

    args = argparse.Namespace(
        adata_path=str(adata_path),
        dir=str(workspace) + os.sep,  # UCE concatenates strings, keep the separator
        species=species,
        model_loc=str(model_dir / "4layer_model.torch"),
        batch_size=int(batch_size),
        spec_chrom_csv_path=str(model_dir / "species_chrom.csv"),
        offset_pkl_path=str(model_dir / "species_offsets.pkl"),
        token_file=str(model_dir / "all_tokens.torch"),
        protein_embeddings_dir=str(model_dir / "protein_embeddings"),
        **UCE_KWARGS,
    )

    sys.path.insert(0, str(UCE_REPO))  # UCE imports its siblings as top-level modules
    try:
        from evaluate import AnndataProcessor

        np.random.seed(0)  # cell sentences are sampled, so pin the draw
        torch.manual_seed(0)
        processor = AnndataProcessor(args, Accelerator(project_dir=str(workspace)))
        processor.preprocess_anndata()
        processor.generate_idxs()
        processor.run_evaluation()
    finally:
        sys.path.remove(str(UCE_REPO))

    written = Path(str(workspace) + os.sep + "input_uce_adata.h5ad")
    if not written.exists():
        raise RuntimeError(f"UCE did not write {written}; see the console output above")
    embedded = ad.read_h5ad(written)
    if EMBEDDING_NAME not in embedded.obsm:
        raise RuntimeError(f"UCE output has no obsm[{EMBEDDING_NAME!r}]")
    return embedded


def reference_path(organism: str, organ: str, n_cells: int) -> Path:
    """Location of the cached reference embeddings for one organism/organ/size."""
    slug = (organ or "all").replace(" ", "_")
    species = SPECIES.get(organism, organism)
    return (
        CACHE_DIR
        / "reference"
        / CENSUS_VERSION
        / f"uce-{species}"
        / f"{slug}-n{n_cells}.npz"
    )


def build_reference(
    organism: str,
    organ: str = "brain",
    n_cells: int = N_REFERENCE_CELLS,
) -> Path:
    """Embed a Census reference once and cache embeddings + labels to disk."""
    organism = organism.strip().replace(" ", "_").lower()
    organ = organ.strip()
    species = resolve_species(organism)
    target = reference_path(organism, organ, n_cells)

    value_filter = "is_primary_data == True"
    if organ:
        value_filter = f"tissue_general == '{organ}' and {value_filter}"

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        obs = (
            census["census_data"][organism]
            .obs.read(
                value_filter=value_filter,
                column_names=["soma_joinid", "cell_type", "tissue_general"],
            )
            .concat()
            .to_pandas()
        )
        if obs.empty:
            raise ValueError(f"No Census cells for {organism!r} in organ {organ!r}")
        # Equal cells per type, so rare cortical types survive in the reference.
        per_type = max(1, n_cells // max(1, obs["cell_type"].nunique()))
        subset = obs.groupby("cell_type", group_keys=False).head(per_type)
        subset = subset.sample(frac=1.0, random_state=0)
        print(f"Reference: {len(subset)} of {len(obs)} Census cells")
        adata = cellxgene_census.get_anndata(
            census,
            organism,
            X_name="raw",
            obs_coords=[int(value) for value in subset["soma_joinid"]],
            obs_column_names=["soma_joinid", "cell_type", "tissue_general"],
            var_column_names=["feature_name"],
        )

    adata.var_names = adata.var["feature_name"].astype(str)  # UCE needs symbols
    adata.var_names_make_unique()

    workspace = CACHE_DIR / "tmp" / f"ref-{os.getpid()}-{int(time.time())}"
    try:
        embedded = embed_anndata(adata, species, workspace)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        embeddings=np.asarray(embedded.obsm[EMBEDDING_NAME], dtype=np.float32),
        cell_type=np.asarray(embedded.obs["cell_type"].astype(str), dtype=str),
        tissue_general=np.asarray(
            embedded.obs["tissue_general"].astype(str), dtype=str
        ),
    )
    print(f"Cached reference embeddings at {target}")
    return target


def predict_cell_types(
    profile: str,
    organism: str,
    organ: str,
    total_count: int,
    k: int = K,
    n_reference_cells: int = N_REFERENCE_CELLS,
) -> str:
    def parse_profile(profile_spec: str) -> dict[str, float]:
        """Parse 'GENE=WEIGHT,GENE=WEIGHT' pairs."""
        if "=" not in profile_spec:
            raise ValueError(
                f"Profile spec is not in the expected format GENE=WEIGHT pairs: {profile_spec}"
            )
        pairs = [pair for pair in profile_spec.split(",") if pair.strip()]
        fields = [pair.split("=") for pair in pairs]
        broken = [pair for pair, parts in zip(pairs, fields) if len(parts) != 2]
        if broken:
            raise ValueError(
                f"Malformed profile entries, expected GENE=WEIGHT: {broken}"
            )

        names = [parts[0].strip() for parts in fields]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            print(
                f"NOTE: duplicate gene(s) in profile, keeping the last weight: {duplicates}"
            )

        profile = {}
        for gene, weight in fields:
            symbol = gene.strip()
            try:
                value = float(weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Profile weight for {symbol!r} is not a number: {weight!r}"
                ) from exc
            if not np.isfinite(value):
                raise ValueError(
                    f"Profile weight for {symbol!r} is not finite: {weight!r}"
                )
            profile[symbol] = value

        zero = sorted(gene for gene, weight in profile.items() if weight == 0)
        if zero:
            print(f"NOTE: dropping zero-weight genes: {zero}")
        profile = {gene: weight for gene, weight in profile.items() if weight != 0}

        if not profile:
            raise ValueError(
                "Profile is empty, provide at least one gene with a non-zero weight"
            )
        return profile

    def build_query_anndata(profile: dict[str, float]) -> ad.AnnData:
        """Build a pseudo-cell indexed by gene symbol, as UCE expects.

        Negative weights cannot be expressed as counts and are clipped with a warning.
        """
        negative = sorted(gene for gene, weight in profile.items() if weight < 0)
        if negative:
            print(
                "WARNING: negative weights are clipped to zero, a pseudo-cell cannot "
                f"represent down-regulated markers: {negative}"
            )

        weights = np.array(
            [max(weight, 0.0) for weight in profile.values()], dtype=np.float32
        )
        if weights.sum() <= 0:
            raise ValueError("All profile weights are <= 0, cannot build a pseudo-cell")

        symbols = list(profile)
        counts = np.round(total_count * weights / weights.sum()).astype(np.float32)
        keep = counts > 0
        if not keep.all():
            dropped = [gene for gene, flag in zip(symbols, keep) if not flag]
            print(
                f"NOTE: weights too small to survive rounding at total_count={total_count}: {dropped}"
            )
        symbols = [gene for gene, flag in zip(symbols, keep) if flag]
        counts = counts[keep]
        if not symbols:
            raise ValueError(
                f"Every profile weight rounds to zero at total_count={total_count}, "
                "raise total_count or the weights"
            )

        obs = pd.DataFrame(
            {"batch": ["unassigned"]}, index=pd.Index(["profile"], name="query")
        )
        var = pd.DataFrame(index=pd.Index(symbols, name="gene"))
        return ad.AnnData(X=sp.csr_matrix(counts.reshape(1, -1)), obs=obs, var=var)

    def load_reference(
        organism: str, organ: str, n_cells: int
    ) -> dict[str, np.ndarray]:
        """Load the cached reference, building it on first use."""
        path = reference_path(organism, organ, n_cells)
        if not path.exists():
            path = build_reference(organism, organ, n_cells)
        with np.load(path) as handle:
            return {key: handle[key] for key in handle.files}

    def top_k_cosine(query: np.ndarray, matrix: np.ndarray, k: int) -> tuple[Any, Any]:
        """Indices and cosine similarities of the k closest reference rows."""
        if query.shape[1] != matrix.shape[1]:
            raise ValueError(
                f"Dimensionality mismatch: query {query.shape} vs reference {matrix.shape}"
            )
        k = int(min(max(k, 1), matrix.shape[0]))

        def unit(block: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            return block / np.where(norms > 0, norms, 1.0)

        scores = unit(matrix) @ unit(query)[0]
        best = np.argsort(-scores)[:k]
        return best, scores[best]

    def vote_cell_types(
        cell_types: np.ndarray,
        tissues: np.ndarray,
        similarities: np.ndarray,
        organ: str | None,
    ) -> str:
        """Plurality-vote cell types over the k neighbours, restricted to one organ.

        Falls back to the global vote when the organ contributes no neighbours.
        """
        mask = np.ones(len(cell_types), dtype=bool)
        if organ:
            mask = tissues == organ
            if not mask.any():
                print(
                    f"NOTE: no neighbours from organ {organ!r}, falling back to global vote"
                )
                mask = np.ones(len(cell_types), dtype=bool)

        counts = pd.Series(cell_types[mask]).value_counts()
        return json.dumps(
            {
                "organ": organ,
                "cell_type": counts.index[0],
                "confidence": round(float(counts.iloc[0] / counts.sum()), 3),
                "n_neighbors": int(counts.sum()),
                "votes": {cell_type: int(votes) for cell_type, votes in counts.items()},
            }
        )

    if total_count < 1:
        raise ValueError(f"total_count must be >= 1, got {total_count!r}")

    organism = organism.strip().replace(" ", "_").lower()
    organ = organ.strip()
    species = resolve_species(organism)
    profile = parse_profile(profile)
    reference = load_reference(organism, organ, n_reference_cells)
    print(f"Reference: {reference['embeddings'].shape[0]} cells")

    query = build_query_anndata(profile)
    print(f"Pseudo-cell built from {query.n_vars} named genes")

    workspace = CACHE_DIR / "tmp" / f"query-{os.getpid()}-{int(time.time())}"
    try:
        embedded = embed_anndata(query, species, workspace, batch_size=1)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    indices, similarities = top_k_cosine(
        np.asarray(embedded.obsm[EMBEDDING_NAME], dtype=np.float32),
        np.asarray(reference["embeddings"], dtype=np.float32),
        k,
    )
    return vote_cell_types(
        reference["cell_type"][indices],
        reference["tissue_general"][indices],
        similarities,
        organ,
    )


cell_typer_uce = dspy.Tool(
    name="cell_typer_uce",
    desc="Identify cell types based on gene expression profile of positive markers using UCE embeddings",
    args={
        "profile": {
            "type": "string",
            "desc": "Gene expression profile of positive markers formatted as 'GENE=WEIGHT,GENE=WEIGHT'",
        },
        "organism": {
            "type": "string",
            "desc": (
                "Most appropriate organism to search for similar gene expression profile in based "
                "on the source profile; UCE supports homo_sapiens, mus_musculus, macaca_mulatta, "
                "macaca_fascicularis, sus_scrofa, microcebus_murinus, danio_rerio, xenopus_tropicalis"
            ),
        },
        "organ": {
            "type": "string",
            "desc": "Organ of interest to be inferred from relevant field. If not available, use empty string for a global vote across all organs",
        },
        "total_count": {
            "type": "integer",
            "desc": "Total number of reads per cell to be inferred from relevant field",
            "default": 20000,
        },
    },
    func=predict_cell_types,
)
