from .generate_latent_gaussian import LTRandomGaussian
from .generate_latent_uniform import LTRandomUniform
from .load_latent import LTLatentLoad

from .preview_latent import LTPreviewLatent
from .reshape_latent import LTReshapeLatent, LTLatentToShape
from .blend_latent import LTBlendLatent
from .latent_op import LTLatentOp
from .concat_latent import LTLatentsConcatenate

from .samplers import LTKSampler

from .param_search import LTNumberRangeGaussian, LTNumberRangeUniform, LTFloatSteps

from .visualize_weights import LTVisualizeWeights, LTListLayers
from .feature_visualization import LTFeatureVisualization, LTActivationAtlas
from .frequency_analysis import LTFrequencyResponse, LTEdgeDetectorAnalysis, LTGaborFit

NODE_CLASS_MAPPINGS = {
    "LTLatentLoad": LTLatentLoad,
    "LTLatentsConcatenate": LTLatentsConcatenate,
    "LTPreviewLatent": LTPreviewLatent,
    "LTGaussianLatent": LTRandomGaussian,
    "LTUniformLatent": LTRandomUniform,
    "LTKSampler": LTKSampler,
    "LTReshapeLatent": LTReshapeLatent,
    "LTLatentToShape": LTLatentToShape,
    "LTBlendLatent": LTBlendLatent,
    "LTLatentOp": LTLatentOp,
    "LTNumberRangeUniform": LTNumberRangeUniform,
    "LTNumberRangeGaussian": LTNumberRangeGaussian,
    "LTVisualizeWeights": LTVisualizeWeights,
    "LTListLayers": LTListLayers,
    "LTFeatureVisualization": LTFeatureVisualization,
    "LTActivationAtlas": LTActivationAtlas,
    "LTFrequencyResponse": LTFrequencyResponse,
    "LTEdgeDetectorAnalysis": LTEdgeDetectorAnalysis,
    "LTGaborFit": LTGaborFit,
} | { f.__name__: f for f in LTFloatSteps }

WEB_DIRECTORY="./web/js"
