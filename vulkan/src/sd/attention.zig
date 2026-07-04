//! Attention dispatcher — launches the attention.comp shader for self-attention.

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
    N: u32,
    dim: u32,
    num_heads: u32,
    head_dim: u32,
    scale: f32,
};

pipeline: Pipeline,
descriptor_pool: c.VkDescriptorPool,
descriptor_set: c.VkDescriptorSet,

pub fn create(inst: *const Instance, spirv: []const u8) !Attention {
    const bindings = [_]c.VkDescriptorSetLayoutBinding{
        .{ .binding = 0, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT },
        .{ .binding = 1, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT },
        .{ .binding = 2, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT },
        .{ .binding = 3, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT },
    };
    const pipe = try Pipeline.create(inst, spirv, &bindings, @sizeOf(PushConstants));

    // Descriptor pool
    const pool_sizes = [_]c.VkDescriptorPoolSize{
        .{ .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4 },
    };
    var descriptor_pool: c.VkDescriptorPool = undefined;
    const pool_ci = c.VkDescriptorPoolCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_sizes,
    };
    _ = c.vkCreateDescriptorPool(inst.device, &pool_ci, null, &descriptor_pool);

    // Descriptor set
    var descriptor_set: c.VkDescriptorSet = undefined;
    const alloc_info = c.VkDescriptorSetAllocateInfo{
        .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = @ptrCast(&pipe.descriptor_set_layout),
    };
    _ = c.vkAllocateDescriptorSets(inst.device, &alloc_info, &descriptor_set);

    return Attention{
        .pipeline = pipe,
        .descriptor_pool = descriptor_pool,
        .descriptor_set = descriptor_set,
    };
}

pub fn dispatch(
    self: *Attention,
    cmd: vk.VkCommandBuffer,
    q: *const Tensor,
    k: *const Tensor,
    v: *const Tensor,
    o: *Tensor,
    N: u32,
    dim: u32,
    num_heads: u32,
) void {
    const head_dim = dim / num_heads;

    // Write descriptors
    const q_desc = c.VkDescriptorBufferInfo{ .buffer = q.buffer.buffer, .offset = 0, .range = q.buffer.size };
    const k_desc = c.VkDescriptorBufferInfo{ .buffer = k.buffer.buffer, .offset = 0, .range = k.buffer.size };
    const v_desc = c.VkDescriptorBufferInfo{ .buffer = v.buffer.buffer, .offset = 0, .range = v.buffer.size };
    const o_desc = c.VkDescriptorBufferInfo{ .buffer = o.buffer.buffer, .offset = 0, .range = o.buffer.size };

    const writes = [_]c.VkWriteDescriptorSet{
        .{ .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = self.descriptor_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &q_desc },
        .{ .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = self.descriptor_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &k_desc },
        .{ .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = self.descriptor_set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &v_desc },
        .{ .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = self.descriptor_set, .dstBinding = 3, .descriptorCount = 1, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &o_desc },
    };
    c.vkUpdateDescriptorSets(self.pipeline.device, 4, &writes, 0, null);

    // Bind
    c.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline.pipeline);
    c.vkCmdBindDescriptorSets(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline.pipeline_layout, 0, 1, @ptrCast(&self.descriptor_set), 0, null);

    // Push constants
    const pk = PushConstants{
        .N = N,
        .dim = dim,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
    };
    c.vkCmdPushConstants(cmd, self.pipeline.pipeline_layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConstants), @ptrCast(&pk));

    // Dispatch: one workgroup per (position, head)
    // Each workgroup handles head_dim=64 threads
    c.vkCmdDispatch(cmd, N, num_heads, 1);
}

pub fn deinit(self: *Attention) void {
    c.vkDestroyDescriptorPool(self.pipeline.device, self.descriptor_pool, null);
    self.pipeline.deinit();
}

const Attention = @This();
