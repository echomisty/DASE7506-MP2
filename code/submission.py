"""Frozen MP2 prediction interface.

Inference reads only the supplied bank, embeddings, and optional checkpoint.
Reference answers and question IDs are never used as model inputs.
"""

from verifier import predict_choices


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    bank: list of question dictionaries with four candidate texts each.
    arrays['embeddings']: float32 NumPy array of shape [N, 4, 1024].
    checkpoint: pathlib.Path or None, supplied through --checkpoint.
    """
    return predict_choices(bank, arrays, checkpoint)
