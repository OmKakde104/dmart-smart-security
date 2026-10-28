import re
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO

# Optional on non-Windows systems; the app works without a system beep.
try:
    import winsound
except ImportError:
    winsound = None

st.set_page_config(page_title="D-Mart Smart Billing & Security", layout="wide")
st.title("🏬 D-Mart Smart Billing & Security")
st.caption("Barcode billing and camera-based item monitoring demo")

REPORT_DIR = Path("stolen_reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PRICE_LIST = {
    "bottle": 45.00, "backpack": 899.00, "handbag": 1200.00,
    "cup": 99.00, "cell phone": 15000.00, "book": 250.00,
    "scissors": 60.00, "laptop": 45000.00, "suitcase": 2500.00,
}
MONITORED_ITEMS = {"backpack", "handbag", "suitcase", "bottle", "cup", "cell phone", "book", "scissors"}


def beep(frequency=1500, duration=250):
    if winsound is not None:
        try:
            winsound.Beep(frequency, duration)
        except (RuntimeError, OSError):
            pass


def normalize_item(value):
    """Convert supported barcode, QR, or manual values into a product name."""
    value = re.sub(r"\s+", " ", str(value).strip().lower())
    aliases = {
        "phone": "cell phone", "mobile": "cell phone", "cellphone": "cell phone",
        "water bottle": "bottle", "bag": "handbag",
    }
    value = aliases.get(value, value)
    return value if value in PRICE_LIST else None


@st.cache_resource
def load_model(weights_path):
    return YOLO(weights_path)


def open_camera(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        return None
    # Keep only the newest frame so slow inference does not queue stale frames.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def close_camera():
    cap = st.session_state.get("camera")
    if cap is not None:
        cap.release()
    st.session_state.camera = None
    st.session_state.camera_mode = None


def start_camera(mode, source):
    close_camera()
    cap = open_camera(source)
    if cap is None:
        st.session_state.camera_error = "Could not open the camera. Check the camera source, IP address, and permissions."
        return False
    st.session_state.camera = cap
    st.session_state.camera_mode = mode
    st.session_state.camera_error = ""
    return True


def read_frame():
    cap = st.session_state.get("camera")
    if cap is None:
        return None
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.resize(frame, (640, 480))


def scan_codes(frame):
    found = []
    try:
        qr = st.session_state.qr_detector
        value, _, _ = qr.detectAndDecode(frame)
        if value:
            found.append(value)
    except cv2.error:
        pass
    try:
        barcode = st.session_state.barcode_detector
        result = barcode.detectAndDecode(frame)
        # OpenCV versions differ in the number of returned values.
        decoded = result[1] if len(result) > 1 else None
        if isinstance(decoded, str):
            found.append(decoded)
        elif decoded is not None:
            found.extend(v for v in decoded if v)
    except (cv2.error, AttributeError, TypeError):
        pass
    return found


def draw_label(frame, box, text, color):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# Streamlit session state persists between fragment refreshes.
DEFAULTS = {
    "cart": {}, "camera": None, "camera_mode": None, "camera_error": "",
    "first_seen": {}, "missing": {}, "saved_alerts": set(),
    "barcode_detector": None, "qr_detector": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, (dict, set)) else value
if st.session_state.barcode_detector is None:
    try:
        st.session_state.barcode_detector = cv2.barcode.BarcodeDetector()
    except (AttributeError, cv2.error):
        st.session_state.barcode_detector = None
if st.session_state.qr_detector is None:
    st.session_state.qr_detector = cv2.QRCodeDetector()

st.sidebar.header("⚙️ Camera")
st.sidebar.caption("The default model detects common COCO objects. Enter a path to custom .pt weights to use a different model.")
model_weights = st.sidebar.text_input("YOLO weights", "yolov8n.pt")
camera_kind = st.sidebar.selectbox("Camera source", ["Laptop Webcam", "Mobile IP Webcam"])
mobile_ip = st.sidebar.text_input("Mobile camera IP (host:port)", "192.168.1.5:8080") if camera_kind == "Mobile IP Webcam" else ""
camera_source = 0 if camera_kind == "Laptop Webcam" else f"http://{mobile_ip.strip().removeprefix('http://').rstrip('/')}/video"
if st.sidebar.button("🔄 Clear System & Next Customer", use_container_width=True):
    close_camera()
    st.session_state.cart = {}
    st.session_state.first_seen = {}
    st.session_state.missing = {}
    st.session_state.saved_alerts = set()
    st.rerun()

if st.session_state.camera_error:
    st.error(st.session_state.camera_error)

billing_tab, security_tab, reports_tab = st.tabs(["🛒 Billing Counter", "🚨 Security Gate", "📋 Reports"])

with billing_tab:
    st.header("Point of Sale (POS)")
    left, right = st.columns(2)
    with left:
        st.subheader("📸 Barcode / QR camera")
        start_bill, stop_bill = st.columns(2)
        if start_bill.button("▶️ Open Billing Camera", key="start_bill"):
            if start_camera("billing", camera_source):
                st.rerun()
        if stop_bill.button("⏹️ Close Billing Camera", key="stop_bill"):
            close_camera()
            st.rerun()
        if st.session_state.camera_mode == "billing":
            billing_frame_slot = st.empty()
            billing_message = st.empty()

            @st.fragment(run_every=0.35)
            def billing_camera():
                if st.session_state.camera_mode != "billing":
                    return
                frame = read_frame()
                if frame is None:
                    st.warning("No frame received from the camera. Check the connection.")
                    return
                for raw in scan_codes(frame):
                    item = normalize_item(raw)
                    if item:
                        st.session_state.cart[item] = st.session_state.cart.get(item, 0) + 1
                        beep()
                        billing_message.success(f"Added to receipt: {item.upper()}")
                        break
                cv2.rectangle(frame, (100, 100), (540, 380), (255, 255, 0), 2)
                cv2.putText(frame, "PLACE BARCODE / QR HERE", (110, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                billing_frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            billing_camera()
        elif st.session_state.camera_mode == "security":
            st.info("The security camera is running. Stop the security feed before using the billing camera.")

        st.markdown("---")
        st.subheader("⌨️ Manual billing")
        with st.form("manual_billing", clear_on_submit=True):
            manual_item = st.text_input("Product name (e.g. bottle, book, phone)")
            add_manual = st.form_submit_button("➕ Add to bill")
        if add_manual:
            item = normalize_item(manual_item)
            if item:
                st.session_state.cart[item] = st.session_state.cart.get(item, 0) + 1
                beep()
                st.success(f"Added to bill: {item.upper()}")
                st.rerun()
            else:
                st.error("Product not found in the supported price list. Check the product name.")

    with right:
        st.subheader("🧾 Tax Invoice / Receipt")
        if st.session_state.cart:
            total = 0.0
            for item, qty in st.session_state.cart.items():
                price = PRICE_LIST[item]
                line_total = price * qty
                total += line_total
                st.write(f"**{item.upper()}** · Qty {qty} · ₹{price:,.2f} each · **₹{line_total:,.2f}**")
            st.divider()
            st.subheader(f"💰 Total: ₹{total:,.2f}")
        else:
            st.info("Bill khaali hai. Barcode/QR scan karein ya item manually add karein.")

with security_tab:
    st.header("Loss Prevention Camera Monitor")
    st.caption("Default yolov8n.pt detects common COCO objects. Keys/locks require custom trained weights that also include the person class.")
    start_sec, stop_sec = st.columns(2)
    if start_sec.button("▶️ Start Security Feed", key="start_security"):
        if start_camera("security", camera_source):
            st.rerun()
    if stop_sec.button("⏹️ Stop Security Feed", key="stop_security"):
        close_camera()
        st.rerun()
    sec_frame_slot = st.empty()
    alert_slot = st.empty()
    if st.session_state.camera_mode == "billing":
        st.info("The billing camera is running. Stop it before starting the security feed.")

    if st.session_state.camera_mode == "security":
        @st.fragment(run_every=0.60)
        def security_camera():
            if st.session_state.camera_mode != "security":
                return
            frame = read_frame()
            if frame is None:
                alert_slot.warning("No frame received from the camera. Check the connection.")
                return
            try:
                model = load_model(model_weights)
                results = model.predict(frame, imgsz=320, conf=0.40, max_det=30, verbose=False)
            except Exception as exc:
                alert_slot.error(f"YOLO model load/run nahi hua: {exc}")
                return

            people, objects = [], []
            for result in results:
                for box in result.boxes:
                    label = model.names[int(box.cls.item())].lower()
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    if label == "person":
                        people.append(coords)
                    else:
                        # Include every object class supported by the selected model.
                        objects.append((label, coords))

            counts = {}
            visible = []
            for label, (ox1, oy1, ox2, oy2) in objects:
                # Display every model detection; only person-overlapping objects feed the alert logic.
                held = any(
                    ox1 < px2 and ox2 > px1 and oy1 < py2 and oy2 > py1
                    for px1, py1, px2, py2 in people
                )
                visible.append((label, (ox1, oy1, ox2, oy2), held))
                if held:
                    counts[label] = counts.get(label, 0) + 1
            for label, quantity in counts.items():
                st.session_state.first_seen[label] = max(st.session_state.first_seen.get(label, 0), quantity)
                st.session_state.missing[label] = 0

            theft_label = None
            for label, seen_qty in list(st.session_state.first_seen.items()):
                missing = seen_qty > st.session_state.cart.get(label, 0) and counts.get(label, 0) < seen_qty
                st.session_state.missing[label] = st.session_state.missing.get(label, 0) + 1 if missing else 0
                if st.session_state.missing[label] >= 15:
                    theft_label = label
                    break

            if theft_label:
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
                cv2.putText(frame, f"ALERT: UNPAID {theft_label.upper()}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                alert_slot.error(f"Possible unpaid {theft_label.upper()} detected. Please verify manually.")
                # Save one image per alert episode; a persistent alert won't fill the disk.
                if theft_label not in st.session_state.saved_alerts:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_label = re.sub(r"[^a-z0-9_-]", "_", theft_label)
                    filename = REPORT_DIR / f"ALERT_{safe_label}_{timestamp}.jpg"
                    if cv2.imwrite(str(filename), frame):
                        st.session_state.saved_alerts.add(theft_label)
                    beep(2500, 300)
            else:
                st.session_state.saved_alerts.clear()
                for label, coords, held in visible:
                    if not held:
                        draw_label(frame, coords, f"DETECTED {label.upper()}", (255, 180, 0))
                    else:
                        paid = st.session_state.cart.get(label, 0) >= counts.get(label, 0)
                        draw_label(frame, coords, f"{'PAID' if paid else 'CHECK'} {label.upper()}", (0, 200, 0) if paid else (0, 220, 255))
                alert_slot.info("Monitor active. Alerts are only prompts for a human to verify.")
            for coords in people:
                draw_label(frame, coords, "PERSON", (255, 255, 255))
            sec_frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        security_camera()
    elif st.session_state.camera_mode is None:
        alert_slot.info("The security feed is stopped. Select Start Security Feed to begin monitoring.")

with reports_tab:
    st.header("📋 Forensic Evidence Logs")
    st.caption("Saved alert snapshots. AI alerts require human verification.")
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









