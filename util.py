import adapter
import db


def get_all_models():
    sdxl = adapter.SDXLLightning("")
    sd1 = adapter.StableDiffusion("")
    sd3 = adapter.StableDiffusion3("")
    fp = adapter.FluxPro("")
    fs = adapter.FluxSchnell("")

    return (sdxl, sd1, sd3, fp, fs)


# models = get_all_models()
# for i in models:
#     r = db.write_model(i.name, i.description, i.version)
#     print(r)
