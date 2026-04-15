from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import urlretrieve

from PIL import Image


DATASET_URLS = {
    "lena": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "baboon": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
    "fruits": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
    "sudoku": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/sudoku.png",
    "smarties": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/smarties.png",
}


def to_pgm(input_path: Path, output_path: Path) -> None:
    img = Image.open(input_path).convert("L")
    img.save(output_path, format="PPM")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "data" / "dataset"
    raw_dir = dataset_dir / "raw"
    pgm_dir = dataset_dir / "pgm"

    raw_dir.mkdir(parents=True, exist_ok=True)
    pgm_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset images...")
    success = 0
    for name, url in DATASET_URLS.items():
        ext = Path(url).suffix.lower() or ".img"
        raw_file = raw_dir / f"{name}{ext}"
        pgm_file = pgm_dir / f"{name}.pgm"

        print(f"  - {name}: {url}")
        try:
            urlretrieve(url, raw_file)
            to_pgm(raw_file, pgm_file)
            success += 1
        except (HTTPError, URLError, OSError) as ex:
            print(f"    failed: {ex}")

    print(f"Done. Downloaded {success}/{len(DATASET_URLS)} images.")
    if success == 0:
        raise SystemExit("No dataset images were downloaded.")

    print("Dataset written to:")
    print(f"  Raw PNG: {raw_dir}")
    print(f"  PGM:     {pgm_dir}")
    print("Example run command:")
    print("  .\\bin\\atrous.exe 512 512 2 20 16 16 data/dataset/pgm/lena.pgm results/lena_out.pgm")


if __name__ == "__main__":
    main()
