import io
import ast
import base64
import asyncio
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

Image.MAX_IMAGE_PIXELS = None
TILE, CONF_MIN = 192, 0.60
IGNORE = {"background"}

# Load ONNX model safely
try:
    sess = ort.InferenceSession("tomato.onnx", providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    SZ = inp.shape[2] if isinstance(inp.shape[2], int) and inp.shape[2] is not None else 160
    names = ast.literal_eval(sess.get_modelmeta().custom_metadata_map["names"])
    CLASSES = [names[i] for i in range(len(names))]
except Exception as e:
    print(f"Warning: Could not load tomato.onnx. Ensure model file is present. Details: {e}")
    sess = None
    SZ = 160
    CLASSES = []

app = FastAPI(title="Plant Disease Scanner")

def classify(t):
    if not sess:
        return "unknown", 0.0
    img_resized = Image.fromarray(t).resize((SZ, SZ))
    x = np.asarray(img_resized, dtype=np.float32)
    tensor = x.transpose(2, 0, 1)[None] / 255.0
    p = sess.run(None, {inp.name: tensor})[0][0]
    i = int(p.argmax())
    return CLASSES[i], float(p[i])

def scan(im: Image.Image):
    arr = np.asarray(im)
    H, W = arr.shape[:2]
    t = min(TILE, W, H)
    found = []

    # Use non-overlapping grid steps for high-res performance
    step = t

    for y in range(0, H - t + 1, step):
        for x in range(0, W - t + 1, step):
            tile_arr = arr[y:y+t, x:x+t]
            label, conf = classify(tile_arr)
            if label in IGNORE or "healthy" in label.lower() or conf < CONF_MIN:
                continue
            found.append({
                "disease": label,
                "confidence": round(conf, 3),
                "x": int(x),
                "y": int(y),
                "size": int(t)
            })
    return found

PAGE = """<!doctype html>
<html>
<head><title>Plant Disease Scanner</title></head>
<body style="font-family:sans-serif;max-width:800px;margin:2em auto;padding:0 1em;">
<h2>Central Valley Plant Disease Scanner (48MP Pipeline)</h2>
<form action="/scan" method="post" enctype="multipart/form-data">
<input type="file" name="file" accept="image/*" required style="margin-bottom: 1em;">
<br>
<button type="submit" style="padding: 0.5em 1em; background: #2e7d32; color: white; border: none; border-radius: 4px; cursor: pointer;">Scan Image</button>
</form>
{result}
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE.format(result="")

@app.post("/scan", response_class=HTMLResponse)
async def scan_page(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
        im = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file format.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {str(e)}")

    # Run CPU inference in background thread to avoid blocking server
    found = await asyncio.to_thread(scan, im)

    # Generate scaled preview with drawn bounding boxes
    preview = im.copy()
    preview.thumbnail((1600, 1600))
    scale = preview.width / im.width

    draw = ImageDraw.Draw(preview)
    for f in found:
        rx, ry, rt = f['x'] * scale, f['y'] * scale, f['size'] * scale
        draw.rectangle([rx, ry, rx+rt, ry+rt], outline="red", width=3)

    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    rows = "".join(f"<li><b>{f['disease']}</b> ({f['confidence']*100:.1f}%) at X:{f['x']}, Y:{f['y']}</li>" for f in found)

    html = f'''
    <hr style="margin: 2em 0;">
    <h3>Scan Results: {len(found)} flagged regions</h3>
    <div style="margin-bottom: 1.5em;">
        <img src="data:image/jpeg;base64,{b64}" style="max-width:100%;height:auto;border: 1px solid #ccc;">
    </div>
    <ul>{rows if rows else "<li>No diseases detected above confidence threshold.</li>"}</ul>
    <p><a href="/">Scan another image</a></p>
    '''
    return PAGE.format(result=html)

@app.post("/api/scan")
async def scan_api(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
        im = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file format.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {str(e)}")

    found = await asyncio.to_thread(scan, im)
    return JSONResponse({"flagged": found})
