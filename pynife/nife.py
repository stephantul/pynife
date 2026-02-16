from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Router

from pynife.utilities import get_teacher_from_metadata


def load_as_router(name: str, teacher_name: str | None = None) -> SentenceTransformer:
    """Load a SentenceTransformer model from the Hugging Face Hub.

    Args:
        name: The name of the model to load.
        teacher_name: The name of the teacher model. If this is None, it will be inferred from the model's metadata.
            We recommend to leave this as None, because it is easy to get wrong.

    Returns:
        SentenceTransformer: The loaded model.

    Raises:
        ValueError: If the dimensionality of the teacher and student models do not match.

    """
    teacher_name = teacher_name or get_teacher_from_metadata(name, "base_model")
    big_model = SentenceTransformer(teacher_name)
    small_model = SentenceTransformer(name)

    # Ensure that both models have the same dimensionality.
    big_dim = big_model.get_sentence_embedding_dimension()
    small_dim = small_model.get_sentence_embedding_dimension()
    if big_dim != small_dim:
        raise ValueError(
            f"Dimensionality mismatch between teacher ({big_dim}) and student ({small_dim}). "
            "Please check that you have the correct teacher model."
        )

    router = Router.for_query_document(
        query_modules=list(small_model),  # type: ignore  # BOOO
        document_modules=list(big_model),  # type: ignore  # BOOO
        default_route="query",
    )
    return SentenceTransformer(modules=[router])
