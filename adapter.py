import uuid
from typing import Any, Dict, Optional

import replicate
from dotenv import load_dotenv

load_dotenv()

HEIGHT = 640
WIDTH = 640


class ImageGen:
    def __init__(self, image: Any):
        self.image = image
        self.img_id = str(uuid.uuid4())


class TTIModel:
    def __init__(self, prompt: str, name: str, description: str, id: str):
        self.id: str = id
        self.prompt: str = prompt
        self.name: str = name
        self.description: str = description

    def generate_image(self, additional_params: Optional[Dict[str, Any]] = None) -> Any:
        input_params = {"prompt": self.prompt}
        if additional_params:
            input_params.update(additional_params)

        output = replicate.run(
            self.model,
            input=input_params,
        )
        return output


class SDXLLightning(TTIModel):
    def __init__(self, prompt: str):
        super().__init__(
            prompt=prompt,
            name="SDXL Lightning 4-step",
            description="SDXL-Lightning by ByteDance",
            id="bytedance/sdxl-lightning-4step:5f24084160c9089501c1b3545d9be3c27883ae2239b6f412990e82d4a6210f8f",
        )

    def generate_image(self, additional_params: Optional[Dict[str, Any]] = None) -> Any:
        if additional_params is None:
            additional_params = {}
        additional_params.update(
            {
                "scheduler": "K_EULER",
                "height": HEIGHT,
                "width": WIDTH,
            }
        )
        return super().generate_image(additional_params)


class StableDiffusion(TTIModel):
    def __init__(self, prompt: str):
        super().__init__(
            prompt=prompt,
            name="Stable Diffusion",
            description="Original Stable Diffusion",
            id="stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
        )

    def generate_image(self) -> Any:
        additional_params = {
            "scheduler": "K_EULER",
            "height": HEIGHT,
            "width": WIDTH,
        }
        return super().generate_image(additional_params)


class StableDiffusion3(TTIModel):
    def __init__(self, prompt: str):
        super().__init__(
            prompt=prompt,
            name="Stable Diffusion 3",
            description="Stable Diffusion 3",
            id="stability-ai/stable-diffusion-3",
        )

    def generate_image(self) -> Any:
        additional_params = {}
        return super().generate_image(additional_params)


class FluxPro(TTIModel):
    def __init__(self, prompt: str):
        super().__init__(
            prompt=prompt,
            name="Flux Pro",
            description="Flux Pro",
            id="black-forest-labs/flux-pro",
        )

    def generate_image(self) -> Any:
        additional_params = {}
        return super().generate_image(additional_params)


class FluxSchnell(TTIModel):
    def __init__(self, prompt: str):
        super().__init__(
            prompt=prompt,
            name="Flux Schnell",
            description="Flux Schnell",
            id="black-forest-labs/flux-schnell",
        )

    def generate_image(self) -> Any:
        additional_params = {}
        return super().generate_image(additional_params)
