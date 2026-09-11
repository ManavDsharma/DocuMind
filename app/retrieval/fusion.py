from app.retrieval.schemas import RetrievedChunk

RRF_K = 60  # standard constant from the original RRF paper; dampens the impact of rank 1 vs rank 2


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """Fuses multiple ranked result lists by rank position, not raw score.

    Each chunk's fused score is the sum of 1/(RRF_K + rank) across every
    list it appears in (1-indexed rank). A chunk ranked highly in *both*
    vector and BM25 results scores higher than one that only appears in
    one list — this is what makes hybrid search actually better than
    either method alone, rather than just concatenating results.
    """
    fused_scores: dict[str, float] = {}
    chunk_lookup: dict[str, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            chunk_lookup[chunk.chunk_id] = chunk

    ranked_chunk_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)

    return [
        chunk_lookup[cid].model_copy(update={"score": fused_scores[cid]})
        for cid in ranked_chunk_ids[:top_k]
    ]