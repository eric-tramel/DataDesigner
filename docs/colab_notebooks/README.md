# Tutorial Notebooks

These notebooks walk through Data Designer's core workflows, from basic synthetic data generation to structured outputs, seed datasets, and image generation.

## Setup

Download the tutorial bundle from the [latest release assets](https://github.com/NVIDIA-NeMo/DataDesigner/releases/latest/download/data_designer_tutorial.zip), extract it, and launch Jupyter from the extracted directory.

```bash
unzip data_designer_tutorial.zip
cd data_designer_tutorial
uv run jupyter notebook
```

Set an API key for the provider you plan to use before running model-backed cells:

```bash
export NVIDIA_API_KEY="your-api-key-here"
export OPENAI_API_KEY="your-api-key-here"
export OPENROUTER_API_KEY="your-api-key-here"
```

## Tutorial Series

- [The Basics](1-the-basics.ipynb)
- [Structured Outputs, Jinja Expressions, and Conditional Generation](2-structured-outputs-and-jinja-expressions.ipynb)
- [Seeding with an External Dataset](3-seeding-with-a-dataset.ipynb)
- [Providing Images as Context](4-providing-images-as-context.ipynb)
- [Generating Images](5-generating-images.ipynb)
- [Image-to-Image Editing](6-editing-images-with-image-context.ipynb)

## Related Documentation

- [Columns](../concepts/columns.md)
- [Default Model Settings](../concepts/models/default-model-settings.md)
- [Config Builder API](../code_reference/config_builder.md)
