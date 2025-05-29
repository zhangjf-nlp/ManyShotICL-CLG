import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .dependencies_graph.evaluation.logical_form_matcher import LogicalFromStructuralMatcher
from .dependencies_graph.evaluation.qdmr_to_logical_form_tokens import QDMRToQDMRStepTokensConverter
from .evaluation.normal_form.normalized_graph_matcher import NormalizedGraphMatchScorer
from .qdecomp_scripts.eval.evaluate_predictions import format_qdmr