"""Streamlit entrypoint. Run from the project root."""
from hashlib import sha256
from html import escape
import json
import logging
import os
from pathlib import Path

import streamlit as st
from inference import DEFAULT_WEIGHTS, load_detector, detect_image
from media import MediaError, VideoOutput, media_kind, open_image, process_video
from translations import TEXT

APP = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)
st.set_page_config(page_title="Road Damage Detection", page_icon="🛣️", layout="wide")


def release_result():
    result = st.session_state.get("road_result")
    if result and isinstance(result.get("video"), VideoOutput):
        result["video"].cleanup()
    st.session_state.road_result = None


def select_input(data, name):
    release_result()
    st.session_state.road_input = {"data": data, "name": name, "id": sha256(data).hexdigest()}
    st.session_state.road_attempted = False
    st.session_state.road_error = None


def select_example(filename):
    select_input((APP / "examples" / filename).read_bytes(), filename)
    st.session_state.road_upload_epoch += 1
    st.session_state.road_seen_upload = None


def clear_input():
    release_result()
    st.session_state.road_input = None
    st.session_state.road_attempted = False
    st.session_state.road_error = None
    st.session_state.road_seen_upload = None
    st.session_state.road_upload_epoch += 1


def request_retry():
    st.session_state.road_retry_requested = True


for key, value in {"road_input": None, "road_result": None, "road_attempted": False,
                   "road_error": None, "road_upload_epoch": 0, "road_seen_upload": None}.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "road_arabic" not in st.session_state:
    st.session_state.road_arabic = st.session_state.get("road_language", "EN") == "AR"
with st.container(key="language_switch"):
    arabic = st.toggle("العربية", key="road_arabic")
language = "AR" if arabic else "EN"
st.session_state.road_language = language
t = TEXT[language]
rtl = language == "AR"
css = (APP / "style.css").read_text(encoding="utf-8")
css += '\n:root { --upload-copy: "' + t["drop"] + '"; }'
if rtl:
    css += '\n.stMainBlockContainer { direction:rtl; text-align:right; font-family:"Cairo",sans-serif; } h1 {letter-spacing:0!important;} .st-key-input_panel [data-testid="stHorizontalBlock"] {direction:rtl;}'
# Localize the native uploader's presentation without changing its native file-picker behavior.
css += '\n[data-testid="stFileUploaderDropzone"] button {font-size:0!important;} [data-testid="stFileUploaderDropzone"] button div {font-size:0!important;} [data-testid="stFileUploaderDropzone"] button::after {content:"' + t["browse"] + '";font-size:13px;}'
st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)
st.title(t["title"])


@st.cache_resource(show_spinner=False)
def detector_for(path, modified_ns, contract_modified_ns):
    return load_detector(path)


def get_detector():
    weights = Path(os.environ.get("ROAD_MODEL_PATH", str(DEFAULT_WEIGHTS)))
    contract = weights.with_name("inference_contract.json")
    return detector_for(str(weights.resolve()), weights.stat().st_mtime_ns, contract.stat().st_mtime_ns)


def heading(number, title):
    st.markdown(f'<div class="panel-heading"><span class="panel-number">{number}</span><h2>{escape(title)}</h2></div>', unsafe_allow_html=True)


EMPTY_SVG = '''<svg viewBox="0 0 200 200" fill="none" aria-hidden="true"><circle cx="100" cy="100" r="91" fill="#dfecdf"/><path d="M70 36L42 167M130 36L158 167" stroke="#627264" stroke-width="3" stroke-linecap="round"/><path d="M100 42v20m0 20v20m0 20v20m0 18v9" stroke="#91ac94" stroke-width="3" stroke-linecap="round"/><path d="M127 107l-15 13 12 10-19 17" stroke="#ad9baa" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/><circle cx="126" cy="125" r="35" stroke="#627264" stroke-width="2" stroke-dasharray="3 5"/></svg>'''

detect_clicked = False
retry_clicked = st.session_state.pop("road_retry_requested", False)
# A widget interaction can interrupt a running script. Offer an explicit retry,
# rather than silently repeating automatic video work on that rerun.
if (st.session_state.road_input and st.session_state.road_attempted
        and not st.session_state.road_result and not st.session_state.road_error):
    st.session_state.road_error = "interrupted"
with st.container(key="workspace"):
    input_column, output_column = st.columns([0.95, 1.4], gap="large")
    with input_column:
        with st.container(key="input_panel"):
            heading("01", t["input"])
            upload = st.file_uploader(t["upload"], type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov"],
                                      key=f"road_upload_{st.session_state.road_upload_epoch}", label_visibility="collapsed")
            st.markdown(f'<p class="format-line">{t["formats"]}</p><p class="limits">{t["limits"]}</p>', unsafe_allow_html=True)
            if upload is not None:
                content = upload.getvalue()
                upload_id = sha256(content).hexdigest() + upload.name
                if upload_id != st.session_state.road_seen_upload:
                    select_input(content, upload.name)
                    st.session_state.road_seen_upload = upload_id
            elif st.session_state.road_seen_upload is not None:
                clear_input()
                st.rerun()
            current = st.session_state.road_input
            if current:
                try:
                    kind = media_kind(current["name"])
                    if kind == "image":
                        st.image(open_image(current["data"]), width="stretch")
                    else:
                        st.markdown('<div class="video-input">▷<br>' + escape(current["name"]) + '</div>', unsafe_allow_html=True)
                        st.caption(t["video_note"])
                    action, clear = st.columns([2, 1], gap="small")
                    with action:
                        if kind == "image":
                            detect_clicked = st.button(t["detect"], type="primary", width="stretch")
                    with clear:
                        st.button(t["clear"], on_click=clear_input, width="stretch")
                except MediaError as error:
                    st.session_state.road_error = error.code
                    st.session_state.road_attempted = True
                    st.button(t["clear"], on_click=clear_input)
            st.markdown(f'<p class="example-label">{t["examples"]}</p>', unsafe_allow_html=True)
            examples = sorted((APP / "examples").glob("*.jpg"))
            if examples:
                with st.container(key="examples"):
                    for column, example in zip(st.columns(len(examples), gap="small"), examples):
                        with column:
                            st.image(str(example), width="stretch")
                            number = examples.index(example) + 1
                            label = f'{t["scene"]} {number if not rtl else str(number).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))}'
                            st.button(label, key="example_" + example.stem, on_click=select_example,
                                      args=(example.name,), width="stretch")
    with output_column:
        with st.container(key="output_panel"):
            heading("02", t["output"])
            result_area = st.empty()
            retry_area = st.empty()

current = st.session_state.road_input
should_process = False
if current:
    try:
        kind = media_kind(current["name"])
        should_process = detect_clicked or retry_clicked or (kind == "video" and not st.session_state.road_attempted)
    except MediaError:
        pass
if should_process:
    st.session_state.road_attempted = True
    st.session_state.road_error = None
    release_result()
    with result_area.container():
        with st.spinner(t["processing"]):
            try:
                try:
                    detector = get_detector()
                except Exception:
                    logger.exception("Could not load detector")
                    raise MediaError("model") from None
                if kind == "image":
                    annotated, count = detect_image(detector, open_image(current["data"]))
                    st.session_state.road_result = {"kind": "image", "image": annotated, "count": count}
                else:
                    bar = st.progress(0, text=t["processing"])
                    result = process_video(current["data"], Path(current["name"]).suffix, detector,
                                           progress=lambda fraction: bar.progress(fraction, text=t["processing"]))
                    st.session_state.road_result = {"kind": "video", "video": result, "count": result.detections}
                    bar.empty()
            except MediaError as error:
                st.session_state.road_error = error.code
            except Exception:
                logger.exception("Could not process media")
                st.session_state.road_error = "failed"
    retry_area.empty()

with result_area.container():
    result = st.session_state.road_result
    error = st.session_state.road_error
    if error:
        st.warning(t.get(error, t["failed"]))
    elif result:
        if result["kind"] == "image":
            st.image(result["image"], width="stretch")
        else:
            st.video(str(result["video"].path), format="video/mp4", autoplay=True, muted=True)
        if result["count"] == 0:
            st.info(t["no_objects"])
    else:
        hint = t["ready"] if current and not st.session_state.road_error else t["empty"]
        st.markdown('<div class="empty-result">' + EMPTY_SVG + '<h3>' + t["empty_title"] + '</h3><p>' + hint + '</p></div>', unsafe_allow_html=True)

if st.session_state.road_error:
    with retry_area.container():
        st.button(t["retry"], key="road_retry", on_click=request_retry)

st.markdown('<div class="road-footer"><p>' + t["credit"] + ' · ' + t["metric"] + ' · <a href="https://zenodo.org/records/10100129" target="_blank" rel="noopener noreferrer">' + t["dataset"] + '</a></p><p class="disclaimer">' + t["disclaimer"] + '</p></div>', unsafe_allow_html=True)
