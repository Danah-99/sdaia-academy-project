import cv2
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# ===== Settings =====
API_KEY = "YOUR_API_KEY"
VIDEO_PATH = "istockphoto-1212771708-640_adpp_is.mp4"
OUTPUT_PATH = "output.mp4"
EVERY_N_FRAMES = 5

COLORS = {
    "with_mask": (0, 255, 0),
    "without_mask": (0, 0, 255),
    "mask_weared_incorrect": (0, 165, 255),
}

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
).configure(InferenceConfiguration(api_key_transport="header"))

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

frame_count = 0
preds = []

while True:
    ok, frame = cap.read()
    if not ok:
        break

    if frame_count % EVERY_N_FRAMES == 0:
        result = client.run_workflow(
            workspace_name="danahs-workspace-ietsg",
            workflow_id="face-mask-detection-5y6pk-1-yolo26n-t2-vface-mask-detection-5y6pk-1-yolo26n-t2-logic",
            images={"image": frame},
            use_cache=True
        )
        preds = result[0]["predictions"]["predictions"]

    for p in preds:
        x1 = int(p["x"] - p["width"] / 2)
        y1 = int(p["y"] - p["height"] / 2)
        x2 = int(p["x"] + p["width"] / 2)
        y2 = int(p["y"] + p["height"] / 2)
        color = COLORS.get(p["class"], (255, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{p["class"]} {p["confidence"]:.2f}', (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)
    cv2.imshow("Face Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("Done! Saved to", OUTPUT_PATH)

