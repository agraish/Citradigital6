import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = BASE_DIR / "dataset" / "present" / "01_HighQuality_Enhanced.jpg"
CROPS_DIR = BASE_DIR / "crops"

CROPS_DIR.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise SystemExit("Gambar sumber tidak ditemukan.")

# Putar gambar agar tegak
image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# Pilih area tanda tangan
x, y, w, h = cv2.selectROI(
    "Pilih Area Tanda Tangan",
    image,
    fromCenter=False,
    showCrosshair=True
)

cv2.destroyAllWindows()

if w == 0 or h == 0:
    raise SystemExit("Crop dibatalkan.")

# Crop
cropped = image[y:y+h, x:x+w]

# Simpan ke folder crops
output_path = CROPS_DIR / "01_signature_crop.jpg"

success = cv2.imwrite(str(output_path), cropped)

if not success:
    raise SystemExit("Gagal menyimpan hasil crop.")