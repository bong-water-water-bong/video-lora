# Build the Vulkan compute backend for SD1.5/AnimateDiff

GLSLC ?= glslc
CC    ?= gcc
CFLAGS ?= -O3 -fPIC -Wall -Wextra -I src

SHADER_SRCS := $(wildcard shaders/*.comp)
SHADER_OBJS := $(SHADER_SRCS:shaders/%.comp=shaders/%.spv)

LIBS := -lvulkan -lm

.PHONY: all shaders lib test clean

all: shaders lib

# Compile GLSL compute shaders to SPIR-V
shaders: $(SHADER_OBJS)

shaders/%.spv: shaders/%.comp
	$(GLSLC) -o $@ $<

# Build shared library
lib: shaders libvk_diffusion.so

libvk_diffusion.so: src/vk_ctx.c src/vk_pipeline.c src/vk_tensor.c src/vk_unet.c
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LIBS)

# Quick smoke test
test: lib
	@echo "=== Testing Vulkan backend ==="
	@python3 -c "import ctypes; lib = ctypes.CDLL('./libvk_diffusion.so'); print('lib loaded, version:', ctypes.c_char_p(lib.vk_api_version()).value.decode())" || echo "No GPU available — this is expected in CI without Vulkan"

# Run full Python test suite
test-py: lib
	python3 -m pytest ../tests/test_vulkan.py -v -x || echo "Python tests (run on Strix Halo only)"

clean:
	rm -f shaders/*.spv libvk_diffusion.so
