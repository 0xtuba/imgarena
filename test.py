from adapter import *

prompt = "An underwater scene of a vibrant coral reef teeming with life. Schools of tropical fish dart between the coral, while a curious octopus changes colors to blend with its surroundings. Shafts of sunlight penetrate the crystal-clear water."

sdxl = SDXLLightning(prompt)
sd = StableDiffusion(prompt)
sd3 = StableDiffusion3(prompt)
fp = FluxPro(prompt)
fs = FluxSchnell(prompt)

# Generate images
# sdxl_output = sdxl.generate_image()
# sd_output = sd.generate_image()
sd3_output = sd3.generate_image()
# fp_output = fp.generate_image()
# fs_output = fs.generate_image()

# print(fp_output)
# print(sd_output)
print(sd3_output)
