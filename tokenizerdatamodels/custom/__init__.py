try:
    import huggingface_hub
except ImportError:
    raise ImportError(
        "The 'huggingface-hub' package is required for this module. "
        "Please reinstall the 'tokenizer-datamodels' package with the 'custom' extra: "
        "'pip install tokenizer-datamodels[custom]'."
    )
