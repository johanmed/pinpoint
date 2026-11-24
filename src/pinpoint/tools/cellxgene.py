"""Module with CellxGene tools for Pinpoint"""

import json
import urllib.request
from pathlib import Path
from typing import Any

import anndata as ad
import cellxgene_census
import cellxgene_census.experimental
import dspy
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scvi

EMBEDDING_NAME = "scvi"
CENSUS_VERSION = "2025-01-30"
K = 5
NPROBE = 10
MEMORY_GiB = 2


def predict_cell_types(
    profile: str,
    organism: str,
) -> str:
    """Plurality-vote cell type prediction per query cell, with confidence scores."""

    def parse_profile(profile_spec: str) -> dict[str, float]:
        """Parse 'GENE=WEIGHT,GENE=WEIGHT' pairs."""
        if "=" in profile_spec:
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
                    f"NOTE: duplicate gene(s) in profile spec, keeping the last weight: {duplicates}"
                )
            raw = {parts[0]: parts[1] for parts in fields}
        else:
            raise ValueError(
                f"Profile spec is not in the expected format GENE=WEIGHT pairs: {profile_spec}"
            )

        profile = {}
        for gene, weight in raw.items():
            symbol = str(gene).strip()
            if not symbol:
                continue
            if symbol in profile:
                print(
                    f"NOTE: duplicate gene {symbol!r} in profile, keeping the last weight"
                )
            try:
                profile[symbol] = float(weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Profile weight for {symbol!r} is not a number: {weight!r}"
                ) from exc

        zero = sorted(gene for gene, weight in profile.items() if weight == 0)
        if zero:
            print(f"NOTE: dropping zero-weight genes: {zero}")
        profile = {gene: weight for gene, weight in profile.items() if weight != 0}

        if not profile:
            raise ValueError(
                "Profile is empty, provide at least one gene with a non-zero weight"
            )
        return profile

    def fetch_embedding_metadata(census_version: str, organism: str) -> dict[str, Any]:
        """Fetch the Census embedding metadata (model link, dimensionality) once."""
        return cellxgene_census.experimental.get_embedding_metadata_by_name(
            embedding_name=EMBEDDING_NAME,
            organism=organism,
            census_version=census_version,
        )

    def get_model_dir(
        census_version: str,
        organism: str,
        embedding_metadata: dict[str, Any],
        model_dir: Path = Path.home() / ".cache" / f"census-{EMBEDDING_NAME}",
    ) -> Path:
        """Return the local scVI model directory, downloading the model if needed."""

        def s3_to_https(link: str, region: str = "us-west-2") -> str:
            """Translate an s3:// URI into a virtual-hosted-style https URL; pass through others."""
            if not link.startswith("s3://"):
                return link
            bucket, _, key = link[len("s3://") :].partition("/")
            return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"

        organism_dir = model_dir / organism / census_version
        model_path = organism_dir / "model.pt"
        if model_path.exists():
            print(f"Using cached scVI model at {model_path}")
            return organism_dir

        link = s3_to_https(embedding_metadata.get("model_link"))
        if not link:
            raise ValueError(
                f"No scVI model is published for census version {census_version!r}, "
                "pin a version that ships maintained embeddings"
            )
        organism_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading scVI model from {link}")
        partial = model_path.with_name("model.pt.part")
        try:
            urllib.request.urlretrieve(link, partial)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        partial.rename(model_path)
        return organism_dir

    def build_symbol_map(var_table: pd.DataFrame) -> dict[str, str]:
        """Map gene symbols to Ensembl IDs, preferring the most observed gene per symbol."""
        table = var_table.dropna(subset=["feature_name"])
        if "nnz" in table.columns:
            table = table.sort_values("nnz", ascending=False, kind="stable")

        ambiguous = table["feature_name"][table["feature_name"].duplicated()].unique()
        if len(ambiguous):
            print(
                f"NOTE: {len(ambiguous)} ambiguous gene symbols, keeping the most observed copy"
            )

        # drop_duplicates(keep='first') rather than dict(zip(...)), which would keep the last
        table = table.drop_duplicates(subset="feature_name", keep="first")
        return dict(zip(table["feature_name"], table["feature_id"]))

    def resolve_genes(
        profile: dict[str, float],
        symbol_map: dict[str, str],
    ) -> tuple[list[str], list[str], np.ndarray, list[str]]:
        """Resolve profile genes to Ensembl IDs, falling back to a case-insensitive match.
        Returns the resolved symbols alongside their IDs so callers never have to
        realign against the original profile, which may contain unresolved genes.
        """
        upper_map = {
            symbol.upper(): feature_id for symbol, feature_id in symbol_map.items()
        }
        symbols = []
        gene_ids = []
        weights = []
        missing = []
        for gene, weight in profile.items():
            feature_id = symbol_map.get(gene) or upper_map.get(gene.upper())
            if feature_id is None:
                missing.append(gene)
                continue
            symbols.append(gene)
            gene_ids.append(feature_id)
            weights.append(weight)
        return symbols, gene_ids, np.array(weights, dtype=np.float32), missing

    def build_query_anndata(
        profile: dict[str, float],
        symbol_map: dict[str, str],
    ) -> ad.AnnData:
        """Build a pseudo-cell AnnData from a {gene symbol: weight} profile.
        Counts are expected to already be normalized and indexed by Ensembl ID
        as the scVI model expects. Negative weights cannot be expressed as counts
        and are clipped with a warning.
        """
        symbols, gene_ids, weights, missing = resolve_genes(profile, symbol_map)
        if missing:
            print(
                f"WARNING: genes unknown to Census and dropped from the profile: {missing}"
            )
        if not gene_ids:
            raise ValueError(
                f"None of the profile genes map to Census genes: {sorted(profile)}"
            )

        negative = sorted(gene for gene, weight in zip(symbols, weights) if weight < 0)
        if negative:
            print(
                "WARNING: negative weights are clipped to zero, a pseudo-cell cannot "
                f"represent down-regulated markers: {negative}"
            )

        positive = np.clip(weights, 0.0, None)
        if positive.sum() <= 0:
            raise ValueError(
                "All resolved profile weights are <= 0, cannot build a pseudo-cell"
            )

        if len(set(gene_ids)) != len(gene_ids):
            # distinct symbols can alias to one Ensembl ID; duplicate var_names would
            # break the scVI gene alignment, so pool their weights instead
            pooled: dict[str, float] = {}
            for feature_id, weight in zip(gene_ids, positive):
                pooled[feature_id] = pooled.get(feature_id, 0.0) + float(weight)
            gene_ids = sorted(pooled)
            positive = np.array(
                [pooled[feature_id] for feature_id in gene_ids], dtype=np.float32
            )
            print(
                "NOTE: multiple profile genes map to the same Ensembl ID, weights pooled"
            )

        counts = np.round(positive / positive.sum()).astype(np.float32)
        var = pd.DataFrame(
            {"feature_id": gene_ids},
            index=pd.Index(gene_ids, name="feature_id"),
        )
        obs = pd.DataFrame(
            {"batch": ["unassigned"]}, index=pd.Index(["profile"], name="query")
        )
        return ad.AnnData(X=sp.csr_matrix(counts.reshape(1, -1)), obs=obs, var=var)

    def encode_profile(adata: ad.AnnData, model_dir: Path) -> ad.AnnData:
        """Project a query AnnData into the Census scVI latent space.
        prepare_query_anndata pads/drops genes to match the model vocabulary, so
        profile genes outside the vocabulary simply contribute zero to the encoding.
        """
        profile_genes = set(map(str, adata.var_names))
        scvi.model.SCVI.prepare_query_anndata(adata, str(model_dir))

        kept_genes = set(map(str, adata.var_names))
        dropped = sorted(profile_genes - kept_genes)
        if dropped:
            print(
                f"WARNING: genes outside the scVI model vocabulary, dropped: {dropped}"
            )
        if not profile_genes & kept_genes:
            raise ValueError(
                "None of the profile genes are in the scVI model vocabulary"
            )

        try:
            vae = scvi.model.SCVI.load_query_data(adata, str(model_dir))
        except KeyError as exc:
            raise RuntimeError(
                "The scVI model requires an obs covariate that the query is missing "
                f"(obs columns: {list(adata.obs.columns)}). The model and census version "
                "may be mismatched."
            ) from exc

        vae.is_trained = True  # plain forward pass, no retraining
        adata.obsm[EMBEDDING_NAME] = vae.get_latent_representation()
        return adata

    def find_neighbors(
        query: ad.AnnData,
        organism: str,
        census_version: str,
        embedding_metadata: dict[str, Any],
        k: int = K,
        nprobe: int = NPROBE,
        memory_gib: int = MEMORY_GiB,
    ) -> Any:
        """Search the Census vector index for the k nearest cells to each query cell."""

        expected = embedding_metadata["n_features"]
        actual = query.obsm[EMBEDDING_NAME].shape[1]
        if actual != expected:
            raise ValueError(
                f"Query embedding has {actual} features but the Census index expects {expected}, "
                "the scVI model does not match the pinned census version"
            )

        return cellxgene_census.experimental.find_nearest_obs(
            EMBEDDING_NAME,
            organism,
            census_version,
            query=query,
            k=k,
            nprobe=nprobe,
            memory_GiB=memory_gib,
            embedding_metadata=embedding_metadata,
        )

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        organism = organism.strip().replace(" ", "_").lower()
        var_table = cellxgene_census.get_var(
            census,
            organism,
            column_names=["feature_id", "feature_name", "nnz"],
        )
        symbol_map = build_symbol_map(var_table)
        profile = parse_profile(profile)
        query = build_query_anndata(profile, symbol_map)
        print(f"Pseudo-cell built from {query.n_vars} matched Census genes")

        embedding_metadata = fetch_embedding_metadata(CENSUS_VERSION, organism)
        model_dir = get_model_dir(
            CENSUS_VERSION,
            organism,
            embedding_metadata,
        )

        query = encode_profile(query, model_dir)
        print(f"Encoded into the scVI latent space: {query.obsm[EMBEDDING_NAME].shape}")

        neighbors = find_neighbors(
            query,
            organism,
            CENSUS_VERSION,
            embedding_metadata,
        )
        predictions = cellxgene_census.experimental.predict_obs_metadata(
            organism,
            CENSUS_VERSION,
            neighbors,
            ["cell_type", "tissue_general"],
            experiment=census["census_data"][organism],
        )
    return json.dumps(predictions.to_dict())


cell_typer = dspy.Tool(
    name="cell_typer",
    desc="Identify cell types based on gene expression profile of positive markers",
    args={
        "profile": {
            "type": "string",
            "desc": "Gene expression profile of positive markers formatted as 'GENE=WEIGHT,GENE=WEIGHT'",
        },
        "organism": {
            "type": "string",
            "desc": "Most appropriate organism to search for similar gene expression profile in based on the source profile, must be either 'homo_sapiens' or 'mus_musculus'",
        },
    },
    func=predict_cell_types,
)
