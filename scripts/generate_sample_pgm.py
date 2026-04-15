from pathlib import Path


def generate(width: int, height: int) -> bytes:
    data = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            # Simple deterministic pattern for visual validation.
            v = (x * 3 + y * 5) % 256
            data[y * width + x] = v
    return bytes(data)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    width, height = 1024, 768
    output = data_dir / "sample_input.pgm"

    payload = generate(width, height)
    with output.open("wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        f.write(payload)

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
