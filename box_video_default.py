# y11_detect_video_embed_log_dual.py
import os, cv2, numpy as np, torch, csv
from ultralytics import YOLO

INPUT_VIDEO   = r"./resources/test/videos/Q4_side_510-540.mp4"
IMGSZ = 1536
CONF  = 0.25
IOU   = 0.7
DEVICE = "0" if torch.cuda.is_available() else "cpu"
USE_HALF = (DEVICE != "cpu")
PRINT_PER_BOX = False

CLASSES_KEEP = None
CUSTOM_NAMES = None

CUSTOM_MODEL_PATH = r"models/v6/best.pt"
OUT_DIR = r"./results"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_CUSTOM_MP4 = os.path.join(OUT_DIR, "output_custom.mp4")
OUTPUT_CUSTOM_CSV = os.path.join(OUT_DIR, "detect_log_custom.csv")

DEFAULT_WEIGHT = "yolo11x.pt"
OUTPUT_DEF_MP4 = os.path.join(OUT_DIR, "output_y11x.mp4")
OUTPUT_DEF_CSV = os.path.join(OUT_DIR, "detect_log_y11n.csv")

def color_for_id(cid: int):
    rng = np.random.RandomState(cid * 123457)
    return int(rng.randint(64, 255)), int(rng.randint(64, 255)), int(rng.randint(64, 255))

def draw_detections(frame, boxes, clses, confs, names, show_conf=True):
    for b, c, s in zip(boxes, clses, confs):
        x1, y1, x2, y2 = map(int, b)
        color = color_for_id(int(c))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = names.get(int(c), str(int(c)))
        if show_conf:
            label = f"{label} {float(s):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(0, y1 - 8)
        cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 4), color, -1)
        cv2.putText(frame, label, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    return frame

def run_once(model_id, input_video, output_video, log_csv, imgsz, conf, iou,
             classes_keep=None, custom_names=None, device="cpu", use_half=False):
    assert os.path.exists(input_video), f"The video does not exist {input_video}"
    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
    if log_csv:
        os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)

    model = YOLO(model_id)
    names = model.names if custom_names is None else {i: n for i, n in enumerate(custom_names)}
    if not isinstance(names, dict):
        names = {i: n for i, n in enumerate(names)}

    cap = cv2.VideoCapture(input_video)
    assert cap.isOpened(), f"Cannot open video {input_video}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    assert writer.isOpened(), f"Cannot create video {output_video}"

    csv_writer = None
    if log_csv:
        csv_f = open(log_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(["frame", "count", "summary", "details"])

    print(f"[INFO] model={model_id} device={device} half={use_half} imgsz={imgsz} conf={conf} iou={iou}")
    print(f"[INFO] Input: {input_video} -> Output: {output_video}")

    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(
            source=frame, imgsz=imgsz, conf=conf, iou=iou,
            device=device, half=use_half, verbose=False
        )[0]

        boxes = np.empty((0,4)); clses = np.empty((0,), dtype=int); confs = np.empty((0,))
        if res.boxes is not None and len(res.boxes):
            boxes = res.boxes.xyxy.cpu().numpy()
            clses = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()

            if classes_keep is not None:
                keep = np.isin(clses, np.array(classes_keep, dtype=int))
                boxes, clses, confs = boxes[keep], clses[keep], confs[keep]

        if clses.size:
            unique, counts = np.unique(clses, return_counts=True)
            summary = ", ".join(f"{names.get(int(c),'?')}={int(n)}" for c, n in zip(unique, counts))
        else:
            summary = "none"

        details = ""
        if PRINT_PER_BOX and clses.size:
            lines = []
            for (x1,y1,x2,y2), c, s in zip(boxes, clses, confs):
                lines.append(f"{names.get(int(c),'?')}({s:.2f})@[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
            details = " | ".join(lines)

        if csv_writer is not None:
            csv_writer.writerow([fid, len(clses), summary, details])

        frame = draw_detections(frame, boxes, clses, confs, names, show_conf=True)
        writer.write(frame)

        fid += 1
        if fid % 50 == 0:
            print(f"[INFO] Processed {fid} frames")

    cap.release()
    writer.release()
    if csv_writer is not None:
        csv_f.close()
        print(f"[INFO] log saved: {log_csv}")
    print("[INFO] Done.\n")

def main():
    assert os.path.exists(CUSTOM_MODEL_PATH), f"The model does not exist {CUSTOM_MODEL_PATH}"
    run_once(
        model_id=CUSTOM_MODEL_PATH,
        input_video=INPUT_VIDEO,
        output_video=OUTPUT_CUSTOM_MP4,
        log_csv=OUTPUT_CUSTOM_CSV,
        imgsz=IMGSZ, conf=CONF, iou=IOU,
        classes_keep=CLASSES_KEEP, custom_names=CUSTOM_NAMES,
        device=DEVICE, use_half=USE_HALF
    )

    run_once(
        model_id=DEFAULT_WEIGHT,
        input_video=INPUT_VIDEO,
        output_video=OUTPUT_DEF_MP4,
        log_csv=OUTPUT_DEF_CSV,
        imgsz=IMGSZ, conf=CONF, iou=IOU,
        classes_keep=CLASSES_KEEP, custom_names=None,
        device=DEVICE, use_half=USE_HALF
    )

if __name__ == "__main__":
    main()