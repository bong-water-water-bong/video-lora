//! LoRA merge dispatcher — fuses LoRA weights into base weights on-device.

const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const vk = @import("../vulkan/vk.zig");
const Instance = @import("../vulkan/instance.zig").Instance;
const Pipeline = @import("../vulkan/pipeline.zig").Pipeline;
const Buffer = @import("../vulkan/buffer.zig").Buffer;
const Tensor = @import("../vulkan/buffer.zig").Tensor;

const PushConstants = extern struct {
    out_dim: u32,
    in_dim: u32,
    rank: u32,
    alpha: f32,
};

pipeline: Pipeline,
descriptor_pool: c.VkDescriptorPool,
descriptor_set: c.VkDescriptorSet,

pub fn create(inst: *const Instance, spirv: []const u8) !LoraFuser {
    const bindings = [_]c.VkDescriptorSetLayoutBinding{
        .{ .binding = 0, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT },
        .{ .binding = 1, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT },
        .{ .binding = 2, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT },
    };
    const pipe = try Pipeline.create(inst, spirv, &bindings, @sizeOf(PushConstants));

    const pool_sizes = [_]c.VkDescriptorPoolSize{
        .{ .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3 },
    };
    var descriptor_pool: c.VkDescriptorPool = undefined;
    const pool_ci = c.VkDescriptorPoolCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_sizes,
    };
    _ = c.vkCreateDescriptorPool(inst.device, &pool_ci, null, &descriptor_pool);

    var descriptor_set: c.VkDescriptorSet = undefined;
    const alloc_info = c.VkDescriptorSetAllocateInfo{
        .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = @ptrCast(&pipe.descriptor_set_layout),
    };
    _ = c.vkAllocateDescriptorSets(inst.device, &alloc_info, &descriptor_set);

    return LoraFuser{
        .pipeline = pipe,
        .descriptor_pool = descriptor_pool,
        .descriptor_set = descriptor_set,
    };
}

pub fn dispatch(
    self: *LoraFuser,
    cmd: vk.VkCommandBuffer,
    a_weights: *const Buffer,
    b_weights: *const Buffer,
    base_weights: *Buffer,
    out_dim: u32,
    in_dim: u32,
    rank: u32,
    alpha: f32,
) void {
    const a_desc = c.VkDescriptorBufferInfo{ .buffer = a_weights.buffer, .offset = 0, .range = a_weights.size };
    const b_desc = c.VkDescriptorBufferInfo{ .buffer = b_weights.buffer, .offset = 0, .range = b_weights.size };
    const base_desc = c.VkDescriptorBufferInfo{ .buffer = base_weights.buffer, .offset = 0, .range = base_weights.size };

    const writes = [_]c.VkWriteDescriptorSet{
        .{ .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = self.descriptor_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &a_desc },
        .{ .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = self.descriptor_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &b_desc },
        .{ .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = self.descriptor_set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &base_desc },
    };
    c.vkUpdateDescriptorSets(self.pipeline.device, 3, &writes, 0, null);

    c.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline.pipeline);
    c.vkCmdBindDescriptorSets(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline.pipeline_layout, 0, 1, @ptrCast(&self.descriptor_set), 0, null);

    const pk = PushConstants{ .out_dim = out_dim, .in_dim = in_dim, .rank = rank, .alpha = alpha };
    c.vkCmdPushConstants(cmd, self.pipeline.pipeline_layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConstants), @ptrCast(&pk));

    // One thread per output weight element
    const total = out_dim * in_dim;
    const groups = (total + 63) / 64;
    c.vkCmdDispatch(cmd, groups, 1, 1);
}

pub fn deinit(self: *LoraFuser) void {
    c.vkDestroyDescriptorPool(self.pipeline.device, self.descriptor_pool, null);
    self.pipeline.deinit();
}

const LoraFuser = @This();
