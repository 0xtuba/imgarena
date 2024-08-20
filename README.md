# Imgen Arena

Imgen Arena is an arena-style system for evaluating image generation models. Since it is difficult to build objective benchmarks for image generation, Imgen Arena aims to provide a way to benchmark models against each other, based on user feedback.

## Supported Models

- [SDXL Lightning 4-step](https://replicate.com/bytedance/sdxl-lightning-4step)
- [Stable Diffusion](https://replicate.com/stability-ai/stable-diffusion) 
- [Stable Diffusion 3](https://replicate.com/stability-ai/stable-diffusion-3)
- [Flux Pro](https://replicate.com/black-forest-labs/flux-pro)
- [Flux Schnell](https://replicate.com/black-forest-labs/flux-schnell)
- [Flux Dev](https://replicate.com/black-forest-labs/flux-dev)
- [Kandinsky 2.2](https://replicate.com/ai-forever/kandinsky-2.2)
- [DALL-E 3](https://openai.com/index/dall-e-3/)
- [Midjourney](https://www.midjourney.com/)

## Future Extensions

- Adding new models, particularly [Google Imagen3](https://deepmind.google/technologies/imagen-3/) (waiting for API access) and Llama3 
- Create new categories of prompts, such as "realistic", "portrait", "abstract", specific styles, images with text, etc. and have separate leaderboards for each category.


## Contributing

Please feel free to open an issue if you want to add support for a new model.

If your model is supported on Replicate, please open an issue to add it to the list of supported models. You can make a pull request to add it to the list of supported models in `adapter.py`, for example [this one](https://github.com/0xtuba/imgarena/blob/master/adapter.py#L46-L65). If the model is not supported on Replicate but has an API, you can make a pull request to add it to the adapters like [DALLE3](https://github.com/0xtuba/imgarena/blob/master/adapter.py#L159-L180). 

If the model does not have an API, you will need to generate the images locally based on the prompts in `master.csv`. Once you do that, you can submit an issue with your generated images and I will add them to the database.

## License

This project is licensed under the MIT License.