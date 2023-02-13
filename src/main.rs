use std::ffi::CStr;
use std::fs::File;
use std::io::Read;
use std::mem::size_of;
use std::path::Path;
use std::slice;

use ash::{Device, Entry, Instance};
use ash::extensions::khr::Swapchain;
use ash::vk;
use ash::vk::{BufferUsageFlags, CommandBuffer, CommandPool, DeviceSize, MemoryMapFlags, MemoryPropertyFlags, PhysicalDevice, Pipeline, SharingMode, SurfaceKHR, SwapchainKHR, VertexInputRate};
use glam::{Vec2, Vec3};
use lazy_static::lazy_static;
use memoffset::offset_of;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle};
use shaderc::{CompileOptions, Compiler, ShaderKind};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

#[repr(C)]
pub struct Vertex {
    position: Vec2,
    color: Vec3
}

impl Vertex {

    pub fn new(position: Vec2, color: Vec3) -> Self {
        Self { position, color }
    }

    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .stride(size_of::<Vertex>() as u32)
            .input_rate(VertexInputRate::VERTEX)
    }

    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(Vertex, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, color) as u32)
        ]
    }

}

lazy_static! {
    pub static ref VERTICES: Vec<Vertex> = vec![
        Vertex::new(Vec2::new(-0.5, -0.5), Vec3::new(1.0, 0.0, 1.0)),
        Vertex::new(Vec2::new(0.5, 0.5), Vec3::new(1.0, 1.0, 0.0)),
        Vertex::new(Vec2::new(-0.5, 0.5), Vec3::new(0.0, 1.0, 1.0))
    ];
}

#[inline]
pub fn create_shader_module(device: &Device, path: impl AsRef<Path>, kind: ShaderKind) -> vk::ShaderModule {
    let mut file = File::open(path).unwrap();

    let mut string = String::new();
    file.read_to_string(&mut string).unwrap();

    let compiler = Compiler::new().unwrap();
    let options = CompileOptions::new().unwrap();
    let binary_result = compiler.compile_into_spirv(&string, kind, "shader.glsl", "main",
                                                    Some(&options)).unwrap();

    unsafe {
        let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(binary_result.as_binary());
        device.create_shader_module(&shader_module_create_info, None).unwrap()
    }
}

pub fn create_instance(entry: &Entry, display_handle: RawDisplayHandle) -> Instance {
    unsafe {
        let app_info = vk::ApplicationInfo::default()
            .application_version(vk::make_api_version(0, 1, 0, 0)) // Program version
            .api_version(vk::API_VERSION_1_3); // Vulkan API Version

        let instance_layers = [b"VK_LAYER_KHRONOS_validation\0".as_ptr().cast()];
        let instance_extensions = ash_window::enumerate_required_extensions(display_handle).unwrap();

        let instance_create_info = vk::InstanceCreateInfo::default()
            .enabled_layer_names(&instance_layers)
            .application_info(&app_info) // Info of Application
            .enabled_extension_names(instance_extensions); // Enable extensions

        entry.create_instance(&instance_create_info, None).unwrap()
    }
}

pub fn get_device(instance: &Instance, physical_device: PhysicalDevice) -> Device {
    unsafe {
        let queue_create_info = vk::DeviceQueueCreateInfo::default().queue_family_index(0) // Type of queue
            .queue_priorities(slice::from_ref(&1.0)); // Queue priorities (Alle queue prios müssen zusammen 1 ergeben)
        let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true);
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan13_features);
        let device_extensions = [b"VK_KHR_swapchain\0".as_ptr().cast()];
        let device_create_info = vk::DeviceCreateInfo::default()
            .push_next(&mut features)
            .enabled_extension_names(&device_extensions)// Set enabled device exts
            .queue_create_infos(slice::from_ref(&queue_create_info)); // Infos von allen Queues
        instance.create_device(physical_device, &device_create_info, None).unwrap()
    }
}

pub fn create_swapchain(instance: &Instance, device: &Device, surface: SurfaceKHR, window: &Window) -> (SwapchainKHR, Swapchain) {
    unsafe {
        let swapchain_loader = Swapchain::new(instance, device); // Create swapchain loader
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(2)
            .image_format(vk::Format::B8G8R8A8_UNORM)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(vk::Extent2D { width: window.inner_size().width, height: window.inner_size().height })
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO);
        (swapchain_loader.create_swapchain(&swapchain_create_info, None).unwrap(), swapchain_loader)
    }
}

pub fn create_command_buffer(device: &Device) -> (CommandBuffer, CommandPool) {
    unsafe {
        let command_pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(0);
        let command_pool = device.create_command_pool(&command_pool_create_info, None).unwrap();

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .command_buffer_count(1);
        let command_buffer = device.allocate_command_buffers(&command_buffer_allocate_info).unwrap()[0];
        (command_buffer, command_pool)
    }
}

pub unsafe fn get_memory_type_index(instance: &Instance, physical_device: PhysicalDevice, type_filter: u32, properties: vk::MemoryPropertyFlags) -> u32 {
    let memory_properties = instance.get_physical_device_memory_properties(physical_device);
    for i in 0..memory_properties.memory_type_count as usize {
        if (type_filter & (1 << i)) != 1 && !(memory_properties.memory_types[i].property_flags & properties).is_empty() {
            return i as u32;
        }
    }
    panic!("No support ig... ._.")
}

pub fn create_pipeline(device: &Device, window: &Window, vertex_shader: &str, fragment_shader: &str) -> Pipeline {
    unsafe {
        // Load Shader
        let vertex_shader = create_shader_module(device, vertex_shader, ShaderKind::Vertex);
        let fragment_shader = create_shader_module(device, fragment_shader, ShaderKind::Fragment);

        // Create Pipeline
        let pipeline_vertex_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader)
            .name(CStr::from_ptr(b"main\0".as_ptr().cast()));

        let pipeline_fragment_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader)
            .name(CStr::from_ptr(b"main\0".as_ptr().cast()));

        let binding_description = Vertex::binding_description();
        let attribute_descriptions = Vertex::attribute_descriptions();
        let pipeline_vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(slice::from_ref(&binding_description))
            .vertex_attribute_descriptions(attribute_descriptions.as_slice());


        let pipeline_input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport::default().x(0.0).y(0.0)
            .width(window.inner_size().width as f32)
            .height(window.inner_size().height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::default().extent(vk::Extent2D { width: window.inner_size().width, height: window.inner_size().height });

        let pipeline_viewport_state_create_info = vk::PipelineViewportStateCreateInfo::default().
            scissors(slice::from_ref(&scissor))
            .viewports(slice::from_ref(&viewport));
        let pipeline_rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0);
        let pipeline_multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);
        let pipeline_color_blend_attachment_info = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        let pipeline_color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(slice::from_ref(&pipeline_color_blend_attachment_info));
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_create_info, None).unwrap();

        let mut pipeline_rendering_create_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&[vk::Format::B8G8R8A8_UNORM]);

        let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo::default();

        let stages = &[pipeline_vertex_shader_stage_create_info, pipeline_fragment_shader_stage_create_info];
        let graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut pipeline_rendering_create_info)
            .stages(stages)
            .vertex_input_state(&pipeline_vertex_input_state_create_info)
            .input_assembly_state(&pipeline_input_assembly_state_create_info)
            .viewport_state(&pipeline_viewport_state_create_info)
            .rasterization_state(&pipeline_rasterization_state_create_info)
            .multisample_state(&pipeline_multisample_state_create_info)
            .color_blend_state(&pipeline_color_blend_state_create_info)
            .dynamic_state(&dynamic_state_create_info)
            .base_pipeline_handle(Pipeline::null())
            .layout(pipeline_layout);
        device.create_graphics_pipelines(vk::PipelineCache::null(), slice::from_ref(&graphics_pipeline_create_info),
                                         None).unwrap()[0]
    }
}

fn main() {
    unsafe {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new().build(&event_loop).unwrap();

        let entry = Entry::load().unwrap();
        let instance = create_instance(&entry, window.raw_display_handle());
        let physical_devices = instance.enumerate_physical_devices().unwrap();

        let best_physical_device = physical_devices[0]; // Nicht production ready eigentlich
        let device = get_device(&instance, best_physical_device);
        let queue = device.get_device_queue(0, 0);

        // Surface and Swapchain
        let surface = ash_window::create_surface(&entry, &instance, window.raw_display_handle(),
                                                 window.raw_window_handle(), None).unwrap();
        let (swapchain, swapchain_loader) = create_swapchain(&instance, &device, surface, &window);
        let images = swapchain_loader.get_swapchain_images(swapchain).unwrap();

        let image_views = images.iter().map(|image| {
            let image_view_create_info = vk::ImageViewCreateInfo::default()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::B8G8R8A8_UNORM)
                .components(vk::ComponentMapping::default())
                .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1).level_count(1));

            device.create_image_view(&image_view_create_info, None).unwrap()
        }).collect::<Vec<_>>();

        let (command_buffer, command_pool) = create_command_buffer(&device);

        // Semaphores and Pipeline
        let semaphore = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap();
        let present_semaphore = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap();
        let pipeline = create_pipeline(&device, &window, "shader.vert.glsl", "shader.frag.glsl");

        // Vertex Buffer lol
        let buffer_create_info = vk::BufferCreateInfo::default()
            .usage(BufferUsageFlags::VERTEX_BUFFER)
            .size(DeviceSize::from((size_of::<Vertex>() * VERTICES.len()) as u32))
            .sharing_mode(SharingMode::EXCLUSIVE);

        let vertex_buffer = device.create_buffer(&buffer_create_info, None).unwrap();

        let memory_requirements = device.get_buffer_memory_requirements(vertex_buffer);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(get_memory_type_index(&instance, best_physical_device, memory_requirements.memory_type_bits,
                                                     MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT));
        let allocated_memory = device.allocate_memory(&allocate_info, None).unwrap();
        device.bind_buffer_memory(vertex_buffer, allocated_memory, DeviceSize::from(0 as u64)).unwrap();

        let pointer = device.map_memory(allocated_memory, DeviceSize::from(0 as u64), buffer_create_info.size, MemoryMapFlags::empty()).unwrap();
        std::ptr::copy_nonoverlapping(VERTICES.as_ptr(), pointer.cast(), VERTICES.len());
        device.unmap_memory(allocated_memory);


        // Run Window
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    window_id
                } if window_id == window.id() => {
                    device.free_memory(allocated_memory, None);
                    device.destroy_buffer(vertex_buffer, None);
                    device.free_command_buffers(command_pool, slice::from_ref(&command_buffer));
                    device.destroy_command_pool(command_pool, None);
                    *control_flow = ControlFlow::Exit
                },
                Event::MainEventsCleared => {
                    let (image_index, _) = swapchain_loader.acquire_next_image(swapchain, u64::MAX, semaphore, vk::Fence::null())
                        .unwrap();

                    device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::RELEASE_RESOURCES).unwrap();
                    device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::RELEASE_RESOURCES).unwrap();
                    device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();

                    // Command Buffer ^^

                    let image_memory_barrier = vk::ImageMemoryBarrier::default()
                        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .image(images[image_index as usize])
                        .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1));

                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        slice::from_ref(&image_memory_barrier)
                    );

                    let rendering_attachment_info = vk::RenderingAttachmentInfo::default()
                        .image_view(image_views[image_index as usize])
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0]
                            }
                        });

                    let rendering_info = vk::RenderingInfo::default()
                        .layer_count(1)
                        .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: window.inner_size().width, height: window.inner_size().height }})
                        .color_attachments(slice::from_ref(&rendering_attachment_info));
                    device.cmd_begin_rendering(command_buffer, &rendering_info);

                    device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
                    device.cmd_bind_vertex_buffers(command_buffer, 0, slice::from_ref(&vertex_buffer), slice::from_ref(&DeviceSize::from(0 as u32)));
                    device.cmd_draw(command_buffer, 3, 1, 0, 0);

                    device.cmd_end_rendering(command_buffer);

                    let image_memory_barrier = vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .image(images[image_index as usize])
                        .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1));

                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        slice::from_ref(&image_memory_barrier)
                    );

                    device.end_command_buffer(command_buffer).unwrap();

                    let submit_info = vk::SubmitInfo::default()
                        .wait_semaphores(slice::from_ref(&semaphore))
                        .wait_dst_stage_mask(slice::from_ref(&vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT))
                        .command_buffers(slice::from_ref(&command_buffer))
                        .signal_semaphores(slice::from_ref(&present_semaphore));

                    device.queue_submit(queue, slice::from_ref(&submit_info), vk::Fence::null()).unwrap();

                    let present_info = vk::PresentInfoKHR::default()
                        .image_indices(slice::from_ref(&image_index))
                        .wait_semaphores(slice::from_ref(&present_semaphore))
                        .swapchains(slice::from_ref(&swapchain));
                    swapchain_loader.queue_present(queue, &present_info).unwrap();

                    device.device_wait_idle().unwrap();
                }
                _ => ()
            }
        });
    }
}