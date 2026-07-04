#ifndef VK_DIFFUSION_H
#define VK_DIFFUSION_H

#include <stdint.h>
#include <stdbool.h>

// ---- Vulkan Context ------------------------------------------------

typedef struct {
    void* device;       // VkDevice
    void* physical_device; // VkPhysicalDevice
    void* queue;        // VkQueue
    uint32_t queue_family;
    void* descriptor_pool;
    void* command_pool;
    void* pipeline_cache;
} vk_ctx_t;

// Create/destroy Vulkan context on the first suitable compute-capable GPU
vk_ctx_t* vk_ctx_create(void);
void      vk_ctx_destroy(vk_ctx_t* ctx);

// ---- Tensors (GPU buffers with shape) ------------------------------

typedef struct {
    vk_ctx_t* ctx;
    void*      buffer;  // VkBuffer
    void*      memory;  // VkDeviceMemory
    uint32_t   batch;
    uint32_t   channels;
    uint32_t   height;
    uint32_t   width;
    uint32_t   elements; // total float count
} vk_tensor_t;

vk_tensor_t* vk_tensor_create(vk_ctx_t* ctx, uint32_t b, uint32_t c, uint32_t h, uint32_t w);
void         vk_tensor_destroy(vk_tensor_t* t);
void         vk_tensor_upload(vk_tensor_t* t, const float* data);
void         vk_tensor_download(vk_tensor_t* t, float* data);
void         vk_tensor_fill(vk_tensor_t* t, float value);

// ---- UNet Inference ------------------------------------------------

typedef struct {
    vk_ctx_t* ctx;
    // internal pipelines, parameter buffers
    void* _internal;
} vk_unet_t;

vk_unet_t* vk_unet_create(vk_ctx_t* ctx, const char* weights_path);
void       vk_unet_destroy(vk_unet_t* unet);

// Run one step of the UNet denoising loop
//   latent:  [1, 4, H/8, W/8] noise input
//   text_emb: [1, 77, 768] CLIP text embeddings
//   timestep: scalar timestep
//   output:   [1, 4, H/8, W/8] denoised latent
void vk_unet_step(vk_unet_t* unet, vk_tensor_t* latent,
                  vk_tensor_t* text_emb, float timestep,
                  vk_tensor_t* output);

// ---- LoRA ---------------------------------------------------------

// Fuse a LoRA safetensors file into the loaded UNet weights
int vk_unet_apply_lora(vk_unet_t* unet, const char* lora_path, float weight);

// ---- Utility -------------------------------------------------------

const char* vk_api_version(void);
int         vk_device_count(void);
void        vk_device_info(int device_idx, char* out, int out_size);

#endif // VK_DIFFUSION_H
