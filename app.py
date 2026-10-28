import queue
import re
import threading
from datetime import datetime
from pathlib import Path

import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO

st.set_page_config(page_title="D-Mart Smart Billing & Security", layout="wide")
st.title("🏬 D-Mart Smart Billing & Security")
st.caption("Browser-camera billing and object monitoring demo")

REPORT_DIR = Path("stolen_reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PRICE_LIST = {
    "bottle": 45.00,
    "backpack": 899.00,
    "handbag": 1200.00,
    "cup": 99.00,
    "cell phone": 15000.00,
    "book": 250.00,
    "scissors": 60.00,
    "laptop": 45000.00,
    "suitcase": 2500.00,
}
MODEL_LOCK = threading.Lock()
MODEL_CACHE = {}
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}


def normalize_item(value):
    value = re.sub(r"\s+", " ", str(value).strip().lower())
    aliases = {
        "phone": "cell phone",
        "mobile": "cell phone",
        "cellphone": "cell phone",
        "water bottle": "bottle",
        "bag": "handbag",
    }
    value = aliases.get(value, value)
    return value if value in PRICE_LIST else None


def load_model(weights_path):
    with MODEL_LOCK:
        if weights_path not in MODEL_CACHE:
            MODEL_CACHE[weights_path] = YOLO(weights_path)
        return MODEL_CACHE[weights_path]


def draw_label(image, box, text, color):
    x1, y1, x2, y2 = [int(value) for value in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        text,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


class BrowserCameraProcessor:
    """Process camera frames sent securely from the user's browser."""

    def __init__(self, mode, weights_path, cart, cart_lock, scan_queue=None):
        self.mode = mode
        self.weights_path = weights_path
        self.cart = cart
        self.cart_lock = cart_lock
        self.scan_queue = scan_queue
        self.model = None
        self.frame_number = 0
        self.last_output = None
        self.qr_detector = cv2.QRCodeDetector()
        try:
            self.barcode_detector = cv2.barcode.BarcodeDetector()
        except (AttributeError, cv2.error):
            self.barcode_detector = None
        self.code_latched = False
        self.no_code_frames = 0
        self.first_seen = {}
        self.missing = {}
        self.active_alerts = set()

    def _get_model(self):
        if self.model is None:
            self.model = load_model(self.weights_path)
        return self.model

    def _scan_codes(self, image):
        found = []
        try:
            value, _, _ = self.qr_detector.detectAndDecode(image)
            if value:
                found.append(value)
        except cv2.error:
            pass
        if self.barcode_detector is not None:
            try:
                result = self.barcode_detector.detectAndDecode(image)
                decoded = result[1] if len(result) > 1 else None
                if isinstance(decoded, str):
                    found.append(decoded)
                elif decoded is not None:
                    found.extend(value for value in decoded if value)
            except (cv2.error, AttributeError, TypeError):
                pass
        return found

    def _process_billing(self, image):
        self.frame_number += 1
        if self.frame_number % 5 == 0:
            codes = self._scan_codes(image)
            if codes:
                self.no_code_frames = 0
                if not self.code_latched:
                    raw_value = codes[0].strip()
                    item = normalize_item(raw_value)
                    if self.scan_queue is not None:
                        self.scan_queue.put((raw_value, item))
                    self.code_latched = True
            else:
                self.no_code_frames += 1
                if self.no_code_frames >= 4:
                    self.code_latched = False

        height, width = image.shape[:2]
        x1, y1 = int(width * 0.15), int(height * 0.15)
        x2, y2 = int(width * 0.85), int(height * 0.85)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            image,
            "PLACE BARCODE OR QR HERE",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return image

    def _process_security(self, image):
        self.frame_number += 1
        # Run YOLO on every fourth frame to reduce server load.
        if self.frame_number % 4 != 0 and self.last_output is not None:
            return self.last_output.copy()

        model = self._get_model()
        try:
            with MODEL_LOCK:
                results = model.predict(
                    image,
                    imgsz=320,
                    conf=0.40,
                    max_det=30,
                    verbose=False,
                    device="cpu",
                )
        except Exception as exc:
            cv2.putText(
                image,
                f"Detection error: {str(exc)[:90]}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return image

        people, objects = [], []
        for result in results:
            for box in result.boxes:
                label = str(model.names[int(box.cls.item())]).lower()
                coords = box.xyxy[0].cpu().numpy().astype(int)
                if label == "person":
                    people.append(coords)
                else:
                    objects.append((label, coords))

        frame_counts = {}
        visible_objects = []
        for label, coords in objects:
            ox1, oy1, ox2, oy2 = coords
            held = any(
                ox1 < px2 and ox2 > px1 and oy1 < py2 and oy2 > py1
                for px1, py1, px2, py2 in people
            )
            visible_objects.append((label, coords, held))
            if held:
                frame_counts[label] = frame_counts.get(label, 0) + 1

        for label, quantity in frame_counts.items():
            self.first_seen[label] = max(self.first_seen.get(label, 0), quantity)
            self.missing[label] = 0

        theft_label = None
        for label, seen_quantity in list(self.first_seen.items()):
            unpaid = seen_quantity > self.cart.get(label, 0)
            disappeared = frame_counts.get(label, 0) < seen_quantity
            self.missing[label] = self.missing.get(label, 0) + 1 if unpaid and disappeared else 0
            if self.missing[label] >= 10:
                theft_label = label
                break

        if theft_label:
            height, width = image.shape[:2]
            cv2.rectangle(image, (0, 0), (width - 1, height - 1), (0, 0, 255), 7)
            cv2.putText(
                image,
                f"CHECK: UNPAID {theft_label.upper()}",
                (12, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            if theft_label not in self.active_alerts:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_label = re.sub(r"[^a-z0-9_-]", "_", theft_label)
                filename = REPORT_DIR / f"ALERT_{safe_label}_{timestamp}.jpg"
                cv2.imwrite(str(filename), image)
            self.active_alerts.add(theft_label)
        else:
            self.active_alerts.clear()

        for label, coords, held in visible_objects:
            if not held:
                draw_label(image, coords, f"DETECTED {label.upper()}", (255, 180, 0))
            else:
                with self.cart_lock:
                    paid = self.cart.get(label, 0) >= frame_counts.get(label, 0)
                color = (0, 200, 0) if paid else (0, 220, 255)
                draw_label(image, coords, f"{'PAID' if paid else 'CHECK'} {label.upper()}", color)
        for coords in people:
            draw_label(image, coords, "PERSON", (255, 255, 255))
        self.last_output = image.copy()
        return image

    def __call__(self, video_frame):
        image = video_frame.to_ndarray(format="bgr24")
        image = cv2.resize(image, (640, 480))
        if self.mode == "billing":
            output = self._process_billing(image)
        else:
            output = self._process_security(image)
        return av.VideoFrame.from_ndarray(output, format="bgr24")


def get_processor(key, mode, weights_path, cart, cart_lock, scan_queue=None):
    processors = st.session_state.processors
    processor = processors.get(key)
    if processor is None or processor.weights_path != weights_path:
        processor = BrowserCameraProcessor(mode, weights_path, cart, cart_lock, scan_queue)
        processors[key] = processor
    return processor


if "cart" not in st.session_state:
    st.session_state.cart = {}
if "scan_queue" not in st.session_state:
    st.session_state.scan_queue = queue.Queue()
if "processors" not in st.session_state:
    st.session_state.processors = {}
if "cart_lock" not in st.session_state:
    st.session_state.cart_lock = threading.RLock()

st.sidebar.header("Camera and AI Settings")
st.sidebar.caption("The browser will ask each user to allow camera access when they press Start.")
st.sidebar.caption("The default YOLO model detects common COCO objects. Keys and locks need custom trained weights.")
model_weights = st.sidebar.text_input("YOLO weights file", "yolov8n.pt")
if st.sidebar.button("Clear System and Start Next Customer", use_container_width=True):
    with st.session_state.cart_lock:
        st.session_state.cart.clear()
    st.session_state.processors.clear()
    while not st.session_state.scan_queue.empty():
        try:
            st.session_state.scan_queue.get_nowait()
        except queue.Empty:
            break
    st.rerun()

billing_tab, security_tab, reports_tab = st.tabs(
    ["Billing Counter", "Security Gate", "Reports"]
)

with billing_tab:
    st.header("Point of Sale")
    left, right = st.columns(2)
    with left:
        st.subheader("Browser Barcode and QR Scanner")
        st.write("Press **START** below, then allow camera access in your browser.")
        billing_processor = get_processor(
            "billing", "billing", model_weights, st.session_state.cart, st.session_state.cart_lock, st.session_state.scan_queue
        )
        webrtc_streamer(
            key="billing-browser-camera",
            video_frame_callback=billing_processor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration=RTC_CONFIGURATION,
            async_processing=True,
        )
        st.divider()
        st.subheader("Manual Billing")
        with st.form("manual_billing", clear_on_submit=True):
            manual_item = st.text_input("Product name (for example: bottle, book, phone)")
            add_manual = st.form_submit_button("Add to bill")
        if add_manual:
            item = normalize_item(manual_item)
            if item:
                with st.session_state.cart_lock:
                    st.session_state.cart[item] = st.session_state.cart.get(item, 0) + 1
                st.success(f"Added to bill: {item.upper()}")
            else:
                st.error("Product not found in the supported price list.")

    with right:
        st.subheader("Tax Invoice / Receipt")

        @st.fragment(run_every=0.5)
        def show_receipt():
            scan_queue = st.session_state.scan_queue
            latest_message = None
            while True:
                try:
                    raw_value, item = scan_queue.get_nowait()
                except queue.Empty:
                    break
                if item:
                    with st.session_state.cart_lock:
                        st.session_state.cart[item] = st.session_state.cart.get(item, 0) + 1
                    latest_message = f"Added to receipt: {item.upper()}"
                else:
                    latest_message = f"Scanned code is not in the price list: {raw_value}"
            if latest_message:
                if latest_message.startswith("Added"):
                    st.success(latest_message)
                else:
                    st.warning(latest_message)
            with st.session_state.cart_lock:
                cart = dict(st.session_state.cart)
            if cart:
                total = 0.0
                for item, quantity in cart.items():
                    price = PRICE_LIST[item]
                    line_total = price * quantity
                    total += line_total
                    st.write(
                        f"**{item.upper()}** · Quantity {quantity} · "
                        f"₹{price:,.2f} each · **₹{line_total:,.2f}**"
                    )
                st.divider()
                st.subheader(f"Total: ₹{total:,.2f}")
            else:
                st.info("The receipt is empty. Scan a code or add an item manually.")

        show_receipt()

with security_tab:
    st.header("Loss Prevention Camera Monitor")
    st.caption("Object detection runs on the video sent from the browser camera.")
    st.write("Press **START**, then allow camera access in your browser.")
    security_processor = get_processor(
        "security", "security", model_weights, st.session_state.cart, st.session_state.cart_lock
    )
    webrtc_streamer(
        key="security-browser-camera",
        video_frame_callback=security_processor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=RTC_CONFIGURATION,
        async_processing=True,
    )
    st.info("AI alerts are suggestions only; verify an event manually.")

with reports_tab:
    st.header("Forensic Evidence Logs")
    st.caption("Saved alert snapshots may be cleared when the cloud app is rebuilt.")
    if st.button("Refresh Reports", key="refresh_reports"):
        st.rerun()
    files = sorted(REPORT_DIR.glob("*.jpg"), key=lambda path: path.stat().st_mtime, reverse=True)
    if files:
        for image_path in files:
            image_col, detail_col = st.columns([1, 2])
            with image_col:
                st.image(str(image_path), use_container_width=True)
            with detail_col:
                st.markdown(f"**Case:** {image_path.name}")
                st.warning("Possible event — verify the image and camera context manually.")
            st.divider()
    else:
        st.info("No alert snapshots have been saved yet.")




