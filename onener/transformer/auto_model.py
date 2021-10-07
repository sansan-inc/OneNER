from collections import OrderedDict

from transformers import BertConfig
from transformers import AutoConfig, PretrainedConfig

from .models.baseline_ner import BertBaselineForNer
from .models.mlp_ner import BertMlpForNer
from .models.crf_ner import BertCrfForNer

MODEL_LIST = [
    "AutoModelForBaselineNer",
    "AutoModelForMlpNer",
    "AutoModelForCrfNer"
]

MODEL_FOR_BASELINE_NER_MAPPING = OrderedDict(
    [
        (BertConfig, BertBaselineForNer),
    ]
)

MODEL_FOR_MLP_NER_MAPPING = OrderedDict(
    [
        (BertConfig, BertMlpForNer),
    ]
)

MODEL_FOR_CRF_NER_MAPPING = OrderedDict(
    [
        (BertConfig, BertCrfForNer),
    ]
)


class AutoModelForBaselineNer:
    def __init__(self):
        raise EnvironmentError(
            "AutoModelForTokenClassification is designed to be instantiated "
            "using the `AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForTokenClassification.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        for config_class, model_class in MODEL_FOR_BASELINE_NER_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_MLP_NER_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_BASELINE_NER_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path,
                                                   *model_args,
                                                   config=config,
                                                   **kwargs)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_MLP_NER_MAPPING.keys()),
            )
        )


class AutoModelForMlpNer:
    def __init__(self):
        raise EnvironmentError(
            "AutoModelForTokenClassification is designed to be instantiated "
            "using the `AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForTokenClassification.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        for config_class, model_class in MODEL_FOR_MLP_NER_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_MLP_NER_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_MLP_NER_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path,
                                                   *model_args,
                                                   config=config,
                                                   **kwargs)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_MLP_NER_MAPPING.keys()),
            )
        )


class AutoModelForCrfNer:
    def __init__(self):
        raise EnvironmentError(
            "AutoModelForTokenClassification is designed to be instantiated "
            "using the `AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForTokenClassification.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        for config_class, model_class in MODEL_FOR_CRF_NER_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_CRF_NER_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_CRF_NER_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path,
                                                   *model_args,
                                                   config=config,
                                                   **kwargs)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_CRF_NER_MAPPING.keys()),
            )
        )
