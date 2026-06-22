"""Datasets for MentalRiskES risk assessment.

- ``mental_risk/``: the whole pipeline — load ALL MentalRiskES data (base corpus
  + every IberLEF edition) into transcripts, decide each subject's task, and
  query a model for that task into responses.

DO NOT add explicit __all__ lists here - use auto_export instead.
See src/common/auto_export.py for documentation on how this works.
"""

from ..common.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())
