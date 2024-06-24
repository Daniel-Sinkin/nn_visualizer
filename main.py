from src.graphics_engine import GraphicsEngine


def main():
    node_layouts: list[tuple[int, ...]] = [
        (8, 4, 2, 1),
        (16, 16, 10, 5, 1),
        (2, 8, 20, 4, 4),
        (2, 3, 4, 5, 4, 3, 2, 1, 4, 7),
        (4, 8, 16, 8, 2, 5, 2),
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        (1, 2, 5, 2, 1),
    ]
    for i, layout in enumerate(node_layouts):
        print(f"{i} / {len(node_layouts)} ({i / len(node_layouts)*100:.2f} %)")
        graphics_engine = GraphicsEngine(
            nodes_per_layer=layout, id_=i, record_video=True, max_frames=600
        )
        graphics_engine.run()


if __name__ == "__main__":
    main()
