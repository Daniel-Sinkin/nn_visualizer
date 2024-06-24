from __future__ import annotations

import os
from typing import Optional

import imageio_ffmpeg as ffmpeg
import moderngl as mgl
import numpy as np
import torch
from glm import mat4, vec2, vec3, vec4  # noqa: F401

from .mymath import screenspace_to_ndc

# Suppresses pygame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame as pg  # noqa: E402


class GraphicsEngine:
    def __init__(self, nodes_per_layer=(8, 4, 2, 1), id_=0):
        self.window_size: tuple[float, float] = (1600.0, 900.0)
        self.aspect_ratio: float = self.window_size[0] / self.window_size[1]

        # Setup Pygame
        pg.init()
        self.clock = pg.time.Clock()
        self.delta_time = 0.0

        # Link PyGame with OpenGL
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(
            pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE
        )
        # OpenGL Context
        self.pg_window = pg.display.set_mode(
            self.window_size, flags=pg.OPENGL | pg.DOUBLEBUF
        )
        self.ctx: mgl.Context = mgl.create_context()
        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE)

        # Create screenshots directory
        self.screenshot_dir = "screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)

        self.nodes_per_layer: tuple[int, ...] = nodes_per_layer
        nodes_max: int = max(self.nodes_per_layer)

        self.weights = []
        for l1, l2 in zip(self.nodes_per_layer, self.nodes_per_layer[1:]):
            self.weights.append(torch.randn((l2, l1)) / 10.0)

        xposs = np.linspace(-0.8, 0.8, len(self.nodes_per_layer))
        yposs = []
        for i in self.nodes_per_layer:
            if i == 1:
                yposs.append([0.0])
            else:
                yposs.append(
                    np.linspace(-0.8 * (i / nodes_max), 0.8 * (i / nodes_max), i)
                )

        all_node_ndc_positions = []
        for layer, n in enumerate(self.nodes_per_layer):
            all_node_ndc_positions.append([])
            for j in range(n):
                all_node_ndc_positions[-1].append((xposs[layer], yposs[layer][j]))

        self.node_layers: list[list[Node]] = []
        color_vals = np.linspace(0.3, 0.7, len(self.nodes_per_layer))

        for c_val, node_ndc_positions in zip(
            color_vals,
            all_node_ndc_positions,
        ):
            self.node_layers.append([])
            for position in node_ndc_positions:
                self.node_layers[-1].append(
                    Node(
                        app=self,
                        pos=vec2(position),
                        scale=vec2(
                            min(
                                0.3 / max(self.nodes_per_layer),
                                0.4 / len(self.nodes_per_layer),
                            )
                        )
                        * vec2(1 / self.aspect_ratio, 1.0),
                        outline_width=0.2,
                        color=vec4(vec3(0.6), 1.0),
                    )
                )

        self.lines: dict[tuple[int, int], Line] = {}
        for i, (x1, x2) in enumerate(zip(xposs, xposs[1:])):
            for j1, y1 in enumerate(yposs[i]):
                for j2, y2 in enumerate(yposs[i + 1]):
                    self.lines[(i, j1, j2)] = Line(
                        app=self, start=vec2(x1, y1), end=vec2(x2, y2), thickness=0.005
                    )

        self.yss: list[torch.Tensor] = [torch.rand(self.nodes_per_layer[0]) * 2.0 - 1.0]
        for i in range(len(self.weights)):
            self.yss.append(torch.sigmoid(self.weights[i] @ self.yss[i]))

        self.iterations: int = 0

        self.record_video = True
        self.id = id_

    def render(self) -> None:
        self.ctx.clear(0.15, 0.15, 0.25)

        for node_layer in self.node_layers:
            for node in node_layer:
                node.render()

        for line in self.lines.values():
            line.render()

        pg.display.flip()

    def update(self) -> None:
        for layer, weight_tensor in enumerate(self.weights):
            for i, row in enumerate(weight_tensor):
                for j, col in enumerate(row):
                    self.lines[(layer, j, i)].color = (1 - col) * vec4(
                        1.0, 0.0, 0.5 / 2.0, 1.0
                    ) + col * vec4(0.1, 1.0, 0.5, 1.0)
                    self.lines[(layer, j, i)].thickness = 0.002 + 0.005 * min(
                        abs(float(col)), 1.0
                    )

        for node_layer in self.node_layers:
            for node in node_layer:
                node.update()

        for line in self.lines.values():
            line.update()

        self.yss = [self.yss[0]]
        for i, weight_tensor in enumerate(self.weights):
            self.yss.append(torch.sigmoid(weight_tensor @ self.yss[i]))

        for j, ys in enumerate(self.yss):
            for i, y in enumerate(ys):
                self.node_layers[j][i].value = y.item()

        for weight_tensor in self.weights:
            weight_tensor += 0.4 * (2.0 * torch.rand_like(weight_tensor) - 1.0)
            torch.clamp(weight_tensor, -1.0, 1.0)

    def take_screenshot(self) -> None:
        screen: pg.Surface = pg.display.get_surface()
        screen_surf: pg.Surface = pg.image.fromstring(
            self.ctx.screen.read(components=3), screen.get_size(), "RGB"
        )

        screenshot_path = os.path.join(self.screenshot_dir, f"{self.iterations}.png")
        pg.image.save(screen_surf, screenshot_path)

    def convert_images_to_video(
        self, image_folder: str, output_video: str, fps: int = 60, crf: int = 4
    ) -> None:
        images: list[str] = sorted(
            [img for img in os.listdir(image_folder) if img.endswith(".png")]
        )

        input_pattern = os.path.join(image_folder, "%d.png")

        cmd = (
            ffmpeg.get_ffmpeg_exe(),  # Path to the ffmpeg executable
            "-framerate",
            str(fps),  # Frames per second
            "-i",
            input_pattern,  # Input pattern
            "-c:v",
            "libx264",  # Codec to use for video encoding
            "-crf",
            str(crf),  # Constant Rate Factor for quality control
            "-pix_fmt",
            "yuv420p",  # Pixel format
            output_video,  # Output file
        )

        os.system(" ".join(cmd))

    def iteration(self) -> None:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.is_running = False
            if event.type == pg.MOUSEBUTTONUP:
                xrel, yrel = screenspace_to_ndc(*pg.mouse.get_pos(), *self.window_size)
                print(f"({xrel:.2f}, {yrel:.2f})")

        self.update()
        self.render()
        if self.record_video:
            self.take_screenshot()

    def run(self) -> None:
        self.is_running = True
        while self.is_running:
            self.iteration()
            self.iterations += 1
            self.delta_time = self.clock.tick(60.0) / 1000.0
            if self.iterations >= 600:
                self.is_running = False

        if self.record_video:
            print("Converting screenshots to video...")
            self.convert_images_to_video(
                "screenshots", f"videos/output_video_{self.id}.mp4"
            )
        print("Finished running normally.")


class Node:
    def __init__(
        self,
        app: GraphicsEngine,
        pos: vec2,
        scale: vec2,
        color: vec4,
        outline_width: float,
        color_outline: Optional[vec4] = None,
        value: float = 0.0,
    ):
        self.app: GraphicsEngine = app
        self.ctx: mgl.Context = app.ctx
        self.pos: vec2 = pos
        self.scale: vec2 = scale
        self.color_base = color
        self.color = self.color_base * value
        self.color_outline = color_outline or vec4(0.0, 0.0, 0.0, 1.0)
        self.outline_width = outline_width
        self.value = value

        self.program = self.ctx.program(
            vertex_shader="""
            #version 330

            in vec2 in_position;
            uniform vec2 u_scale;
            uniform vec2 u_pos;

            void main() {
                gl_Position = vec4(u_scale * in_position + u_pos, 0.0, 1.0);
            }
            """,
            fragment_shader="""
            #version 330

            out vec4 fragColor;
            uniform vec4 u_color;

            void main() {
                fragColor = u_color;
            }
            """,
        )
        self.program["u_pos"].write(self.pos)

        circle_vertex_data: np.ndarray[np.float32] = np.array(
            [
                [np.cos(theta), np.sin(theta)]
                for theta in np.linspace(0, 2 * np.pi, 2**6)
            ],
            dtype=np.float32,
        )

        self.vbo: mgl.Buffer = self.ctx.buffer(circle_vertex_data)
        self.vao: mgl.VertexArray = self.ctx.vertex_array(
            self.program, [(self.vbo, "2f", "in_position")]
        )

    def update(self) -> None: ...

    def render(self) -> None:
        # Render normal circle
        self.program["u_color"].value = tuple(
            vec4(0.0, 1.0, 0.0, 1.0) * self.value
            + vec4(1.0, 0.0, 0.0, 1.0) * (1 - self.value)
        )
        self.program["u_scale"].write((1.0 - self.outline_width) * self.scale)
        self.vao.render(mgl.TRIANGLE_FAN)

        # Render outline circle
        self.program["u_color"].value = tuple(self.color_outline)
        self.program["u_scale"].write(1.0 * self.scale)
        self.vao.render(mgl.TRIANGLE_FAN)


class Line:
    def __init__(
        self,
        app: GraphicsEngine,
        start: vec2,
        end: vec2,
        thickness: float,
        color: Optional[vec4] = None,
    ):
        self.app: GraphicsEngine = app
        self.ctx: mgl.Context = app.ctx
        self.start: vec2 = start
        self.end: vec2 = end
        self.thickness: float = thickness
        self.color: vec4 = color or vec4(1.0)

        self.program = self.ctx.program(
            vertex_shader="""
            #version 330

            in vec2 in_position;

            void main() {
                gl_Position = vec4(in_position, 0.0, 1.0);
            }
            """,
            fragment_shader="""
            #version 330

            out vec4 fragColor;
            uniform vec4 u_color;

            void main() {
                fragColor = u_color;
            }
            """,
        )
        self.program["u_color"].write(self.color)

        self.update_quad_vertices()
        self.vbo: mgl.Buffer = self.ctx.buffer(self.quad_vertex_data)
        self.vao: mgl.VertexArray = self.ctx.vertex_array(
            self.program, [(self.vbo, "2f", "in_position")]
        )

    # TODO: Instead of doing this weird thing directly create a single quad and transform that properly.
    def update_quad_vertices(self) -> None:
        direction = self.end - self.start
        length = np.linalg.norm(direction)
        direction /= length

        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        offset = (self.thickness / 2) * normal

        p1 = self.start + offset
        p2 = self.start - offset
        p3 = self.end - offset
        p4 = self.end + offset

        self.quad_vertex_data: np.ndarray[np.float32] = np.array(
            [p1, p2, p3, p3, p4, p1], dtype=np.float32
        )

    def render(self) -> None:
        self.vao.render(mgl.TRIANGLES)

    def update(self) -> None:
        self.program["u_color"].write(self.color / 10.0)
        self.vbo: mgl.Buffer = self.ctx.buffer(self.quad_vertex_data)
        self.vao: mgl.VertexArray = self.ctx.vertex_array(
            self.program, [(self.vbo, "2f", "in_position")]
        )
