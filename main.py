import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import io
import os
import json
import time
import shutil 

# ReportLab Imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Image as PlatypusImage

# Google GenAI Imports
from google import genai
from google.genai import types

# ==============================================================================
# 0. KONFIGURASI DAN INICIALISASI
# ==============================================================================

# API Key Gemini (Best Practice: Gunakan st.secrets atau variabel lingkungan di produksi)
GEMINI_API_KEY = "AIzaSyCMR3w9GRyYXiWmsmNIph_Sx1BzIfm0yfA" 

# Kredensial Roboflow
ROBOFLOW_API_KEY = "2jDbzJsXWACR5parVNix"  
PROJECT_ID = "penilaian-ui-web-ax2rc"
VERSION_NUM = 2
COMBINED_MODEL_ID = f"{PROJECT_ID}/{int(VERSION_NUM)}"

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Analisis & Penilaian UI Otomatis",
    page_icon="🤖",
    layout="wide"
)

# --- Inisialisasi Klien (Best Practice: Caching resource) ---
@st.cache_resource
def load_inference_client(api_key):
    """Membuat klien inferensi Roboflow."""
    try:
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
        return client
    except Exception as e:
        st.sidebar.error(f"Gagal membuat klien Roboflow: {e}")
        return None

@st.cache_resource
def load_gemini_client(api_key):
    """Membuat klien Gemini."""
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.sidebar.error(f"Gagal membuat klien Gemini: {e}")
        return None

roboflow_client = load_inference_client(ROBOFLOW_API_KEY)
gemini_client = load_gemini_client(GEMINI_API_KEY)

# ==============================================================================
# 1. FUNGSI PEMROSESAN GAMBAR (Best Practice: Modularity)
# ==============================================================================

def generate_element_id_map(predictions):
    """
    Menghasilkan peta ID elemen yang konsisten (mis. Button_1, Button_2) 
    dan memetakan ID kembali ke data prediksi mentah. Fungsi ini juga
    memperbarui list predictions dengan key 'element_id'.
    """
    element_counts = {}
    id_map = {}
    
    for i, item in enumerate(predictions):
        class_name = item['class']
        
        if class_name not in element_counts:
            element_counts[class_name] = 0
        element_counts[class_name] += 1
        element_id = f"{class_name}_{element_counts[class_name]}"
        
        # Tambahkan ID ke prediksi mentah (ini penting!)
        item['element_id'] = element_id
        id_map[element_id] = item 
        
    return id_map

def crop_bounding_box(image, prediction):
    """Memotong gambar asli ke area yang ditentukan oleh bounding box."""
    x_center, y_center, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
    x_min = int(x_center - (width / 2))
    y_min = int(y_center - (height / 2))
    x_max = int(x_center + (width / 2))
    y_max = int(y_center + (height / 2))

    # Pastikan koordinat valid
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.width, x_max)
    y_max = min(image.height, y_max)
    
    # Tambahkan sedikit padding (opsional, untuk konteks visual)
    padding = 5 
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image.width, x_max + padding)
    y_max = min(image.height, y_max + padding)
    
    # Pengecekan agar crop tidak gagal jika min > max
    if x_min >= x_max or y_min >= y_max:
        return None # Kembalikan None jika box tidak valid

    cropped_img = image.crop((x_min, y_min, x_max, y_max))
    return cropped_img

# ==============================================================================
# 2. FUNGSI INFERENSI DAN PENILAIAN
# ==============================================================================

def perform_roboflow_detection(original_image, model_id):
    """Menjalankan deteksi Roboflow dan mengembalikan hasil prediksi dan gambar anotasi."""
    temp_input_image_path = "temp_input_image_for_pred.jpeg"
    rgb_image = original_image.convert('RGB')
    
    rgb_image.save(temp_input_image_path, format='JPEG')
    
    result_dict = roboflow_client.infer(temp_input_image_path, model_id=model_id)
    os.remove(temp_input_image_path)
    
    predictions = result_dict.get('predictions', [])
    
    # Generate ID Map dan update predictions (WAJIB dilakukan di sini)
    generate_element_id_map(predictions)
    
    # Pastikan predictions memiliki 'element_id' sebelum menggambar
    annotated_image = draw_annotations(original_image, predictions)
    
    return predictions, annotated_image

def draw_annotations(image, predictions):
    """Menggambar bounding box dan label pada gambar."""
    annotated_image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", size=60) 
    except IOError:
        font = ImageFont.load_default()

    for pred in predictions:
        x_center, y_center, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)
        x_max = x_center + (width / 2)
        y_max = y_center + (height / 2)
        
        element_id = pred.get('element_id', pred['class']) 
        label = f"{element_id} ({pred['confidence']:.0%})"
        color = "red"
        
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
        
        # Tambahkan label (disingkat)
        try:
            text_bbox = draw.textbbox((x_min, y_min), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
             text_width, text_height = draw.textsize(label, font=font)
             
        text_y_min = y_min - text_height - 10
        
        draw.rectangle([x_min, text_y_min, x_min + text_width, y_min], fill=color)
        draw.text((x_min, text_y_min), label, fill="white", font=font)

    return annotated_image

def get_gemini_assessment(image_path, predictions):
    """Mengirim gambar dan detail deteksi ke Gemini untuk penilaian otomatis."""
    
    element_details = []
    for i, pred in enumerate(predictions): 
        element_details.append(
            f"{i+1}. ID: {pred['element_id']}, Class: {pred['class']}, Confidence: {pred['confidence']:.2f}, Box: [x={pred['x']:.0f}, y={pred['y']:.0f}, w={pred['width']:.0f}, h={pred['height']:.0f}]"
        )
    
    detection_summary = "\n".join(element_details)

    # 2. Atur Prompt dan Struktur JSON
    prompt_text = f"""
    Anda adalah pakar Penilaian Desain UI/UX. Tugas Anda adalah menilai tangkapan layar (screenshot) berdasarkan prinsip-prinsip UX berikut:
    1. Relevansi dan Nilai: Apakah desain ini menjawab pain point user?
    2. Usability dan Kejelasan: Apakah user dapat dengan mudah mengerti desain tampilan ini?

    Deteksi objek dalam gambar adalah sebagai berikut:
    ---
    {detection_summary}
    ---

    Berikan penilaian dalam format JSON. JANGAN BERIKAN TEKS PENJELASAN LAIN DI LUAR JSON.
    - Penilaian Umum (General UI): Berikan 3 penilaian LENGKAP untuk Font/Tipografi, Warna/Skema, dan Skala/Hierarki.
    - Penilaian Elemen Dinamis: Berikan Penilaian UI dan Catatan Tambahan untuk setiap elemen yang terdeteksi. Gunakan element ID yang SAMA PERSIS dengan yang tercantum dalam 'detection_summary' (misal: Button_1).

    Gunakan kategori penilaian yang tersedia: Baik Sekali, Baik, Cukup, Kurang, Kurang Sekali. Penjelasan harus lengkap, informatif, dan ringkas dalam satu string.

    Format JSON yang HARUS Anda hasilkan adalah:
    {{
        "penilaian_font": "KATEGORI: [Pilih Kategori]. [Penjelasan lengkap, termasuk alasan kategorisasi dan saran singkat].",
        "penilaian_color": "KATEGORI: [Pilih Kategori]. [Penjelasan lengkap, termasuk alasan kategorisasi dan saran singkat].",
        "penilaian_scale": "KATEGORI: [Pilih Kategori]. [Penjelasan lengkap, termasuk alasan kategorisasi dan saran singkat].",
        "dynamic_elements": [
            {{
                "id": "[element_id]",
                "penilaian_ui": "KATEGORI: [Pilih Kategori]. [Penjelasan lengkap berdasarkan prinsip UX 1 & 2].",
                "catatan_tambahan": "[Catatan spesifik untuk perbaikan elemen ini]."
            }}
            // ... elemen deteksi lainnya
        ]
    }}
    """
    
    # 3. Kirim ke Gemini (Vision Model)
    try:
        image = Image.open(image_path)
        
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image, prompt_text],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        # Cleanup JSON response
        json_text = response.text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:].strip()
        if json_text.endswith("```"):
            json_text = json_text[:-3].strip()

        return json.loads(json_text)

    except Exception as e:
        st.error(f"Gagal mendapatkan penilaian dari Gemini: {e}")
        return None

# ==============================================================================
# 3. FUNGSI PEMBUATAN PDF (Mencakup Bounding Box)
# ==============================================================================

def generate_pdf_report(scores, annotated_image_path, original_image_pil, raw_predictions, image_name):
    """
    Membuat laporan PDF.
    Termasuk gambar bounding box dari elemen yang dinilai.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    elements = []
    
    # Persiapan
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    temp_dir = "temp_crops"
    os.makedirs(temp_dir, exist_ok=True)
    
    raw_prediction_map = {p['element_id']: p for p in raw_predictions if 'element_id' in p} 

    # --- Header Dokumen ---
    elements.append(Paragraph("<b>Laporan Penilaian Desain UI</b>", styles['h1']))
    elements.append(Paragraph(f"Tanggal Laporan: {current_time}", styles['Normal']))
    elements.append(Paragraph(f"Screenshot Asal: {image_name}", styles['Normal']))
    elements.append(Spacer(1, 0.1 * inch))

    # --- Gambar Hasil Deteksi (Perbaikan: Penskalaan Sangat Konservatif) ---
    if annotated_image_path and os.path.exists(annotated_image_path):
        elements.append(Paragraph("<b>Gambar Hasil Deteksi (Anotasi Roboflow)</b>", styles['h2']))
        try:
            # PENGUBAHAN PENTING: Mengurangi batas tinggi maksimum menjadi 7.5 inci (sekitar 540 pt)
            # untuk memastikan gambar muat dengan aman di halaman pertama dan menghindari LayoutError.
            max_w_inch = 6.5
            max_h_inch = 7.5 
            
            max_w_pt = max_w_inch * inch 
            max_h_pt = max_h_inch * inch
            
            img_w_pt = original_image_pil.width
            img_h_pt = original_image_pil.height
            
            scale_w = max_w_pt / img_w_pt
            scale_h = max_h_pt / img_h_pt

            # Gunakan skala yang paling kecil (paling ketat)
            final_scale = min(scale_w, scale_h)
            
            img = PlatypusImage(
                annotated_image_path, 
                width=img_w_pt * final_scale, 
                height=img_h_pt * final_scale
            )
            
            elements.append(img)
        except Exception as e:
            elements.append(Paragraph(f"<i>Gagal menambahkan gambar anotasi ke PDF: {e}</i>", styles['Normal']))
    else:
        elements.append(Paragraph("<i>Gambar hasil deteksi tidak tersedia.</i>", styles['Normal']))
        elements.append(Spacer(1, 0.2 * inch))

    # --- Penilaian Umum ---
    elements.append(Paragraph("<b>Penilaian Umum</b>", styles['h2']))
    elements.append(Spacer(1, 0.1 * inch))

    data_umum = [
        ["Kategori", "Penilaian"],
        # Memastikan teks panjang di-wrap dengan Paragraph
        ["Font/Tipografi", Paragraph(scores.get('penilaian_font', 'Penilaian tidak tersedia dari Gemini'), styles['Normal'])],
        ["Warna/Skema", Paragraph(scores.get('penilaian_color', 'Penilaian tidak tersedia dari Gemini'), styles['Normal'])],
        ["Skala/Hierarki", Paragraph(scores.get('penilaian_scale', 'Penilaian tidak tersedia dari Gemini'), styles['Normal'])]
    ]
    table_umum = Table(data_umum, colWidths=[2*inch, 4.5*inch]) 
    table_umum.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12), ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    elements.append(table_umum)
    elements.append(Spacer(1, 0.2 * inch))

    # --- Penilaian Elemen Dinamis (dengan Bounding Box) ---
    elements.append(Paragraph("<b>Penilaian Elemen</b>", styles['h2']))
    elements.append(Spacer(1, 0.1 * inch))

    final_table_data = [
        ["Nama Elemen", "Box", "Penilaian UI", "Catatan Tambahan"] 
    ]

    dynamic_elements = scores.get('dynamic_elements', [])
    
    for item in dynamic_elements:
        element_id = item.get('id', 'Unknown_Element')
        penilaian_ui = item.get('penilaian_ui', 'Tidak Dinilai/Kosong')
        catatan = item.get('catatan_tambahan', 'Tidak ada catatan khusus.')
        display_name = element_id.replace('_', ' ').title()
        
        # --- Bounding Box Image Cell ---
        box_image_cell = Paragraph("N/A", styles['Normal'])
        if element_id in raw_prediction_map:
            pred = raw_prediction_map[element_id]
            try:
                cropped_img_pil = crop_bounding_box(original_image_pil, pred)
                
                if cropped_img_pil:
                    # PERBAIKAN PENTING: Konversi RGBA ke RGB sebelum menyimpan sebagai JPEG
                    if cropped_img_pil.mode == 'RGBA':
                        cropped_img_pil = cropped_img_pil.convert('RGB')
                        
                    temp_crop_path = os.path.join(temp_dir, f"{element_id}.jpeg")
                    cropped_img_pil.save(temp_crop_path, format='JPEG')
                    
                    max_cell_size = 1 * inch 
                    
                    # Skala gambar agar pas di dalam sel
                    w, h = cropped_img_pil.size
                    scale = min(max_cell_size / w, max_cell_size / h)
                    final_w = w * scale
                    final_h = h * scale
                    
                    box_image_cell = PlatypusImage(temp_crop_path, width=final_w, height=final_h)
                else:
                    box_image_cell = Paragraph("Box Tidak Valid", styles['Normal'])
                
            except Exception as e:
                # Menampilkan error 'Gagal crop: cannot write mode RGBA as JPEG' telah diselesaikan di atas
                box_image_cell = Paragraph(f"Gagal crop: {e}", styles['Normal']) 

        # Append data row
        final_table_data.append([
            Paragraph(f"<b>{display_name}</b>", styles['Normal']),
            box_image_cell, 
            Paragraph(penilaian_ui, styles['Normal']),
            Paragraph(catatan, styles['Normal'])
        ])

    # Buat dan styling tabel dinamis
    if len(final_table_data) > 1:
        # Lebar Kolom: Elemen (1.0), Gambar (1.3), Penilaian UI (2.3), Catatan (2.3). Total 6.9 inch (sesuaikan)
        table_dynamic = Table(final_table_data, colWidths=[1.0*inch, 1.3*inch, 2.3*inch, 2.3*inch], repeatRows=1)
        table_dynamic.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 8), ('BACKGROUND', (0,1), (-1,-1), colors.white),
            ('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        elements.append(table_dynamic)

    # Membangun PDF
    doc.build(elements)
    
    # Cleanup (Best Practice: Selalu hapus file/folder sementara)
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return buffer.getvalue()


# ==============================================================================
# 4. ANTARMUKA UTAMA STREAMLIT
# ==============================================================================

st.header("🤖 Analisis & Penilaian Desain UI Otomatis")

if not roboflow_client or not gemini_client:
    st.error("Gagal menginisialisasi klien API. Mohon cek konfigurasi.")
    st.stop()

col1, col2 = st.columns(2)

# --- Inisialisasi State ---
if 'annotated_image_path' not in st.session_state: st.session_state.annotated_image_path = None
if 'all_scores' not in st.session_state: st.session_state.all_scores = None
if 'detection_results' not in st.session_state: st.session_state.detection_results = None
if 'original_image_pil' not in st.session_state: st.session_state.original_image_pil = None
if 'submitted' not in st.session_state: st.session_state.submitted = False


with col1:
    st.header("1. Unggah Screenshot")
    uploaded_file = st.file_uploader("Pilih file gambar (PNG, JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            # Gunakan st.cache_data untuk caching file upload
            @st.cache_data(show_spinner=False)
            def load_image(file):
                return Image.open(file)
            
            original_image = load_image(uploaded_file)
            st.session_state.original_image_pil = original_image 
            
            # --- TAMPILAN GAMBAR ASLI DIBUAT EXPANDER ---
            with st.expander("Lihat Gambar Asli yang Diunggah"):
                st.image(original_image, use_column_width=True)

        except Exception as e:
            st.error(f"Gagal memuat gambar: {e}")
            st.stop()
        
        # --- Reset State Jika Ada Unggahan Baru ---
        if st.session_state.get('last_upload_name') != uploaded_file.name:
             st.session_state.all_scores = None
             st.session_state.submitted = False 
             st.session_state.detection_results = None
             if st.session_state.annotated_image_path and os.path.exists(st.session_state.annotated_image_path):
                 os.remove(st.session_state.annotated_image_path)
                 st.session_state.annotated_image_path = None
             st.session_state.last_upload_name = uploaded_file.name

        
with col2:
    st.header("2. Hasil Deteksi & Penilaian")
    
    original_image = st.session_state.get('original_image_pil')

    if original_image and COMBINED_MODEL_ID:
        
        # Tampilkan hasil anotasi jika sudah selesai
        if st.session_state.get('submitted') and st.session_state.get('detection_results'):
            predictions = st.session_state.detection_results.get('predictions', [])
            generate_element_id_map(predictions)
            annotated_image = draw_annotations(original_image, predictions)
            st.image(annotated_image, caption="Gambar dengan Deteksi Elemen", use_column_width=True)
            st.success(f"Penilaian otomatis selesai. Ditemukan **{len(predictions)}** elemen.")
        
        
        if st.button("Mulai Deteksi & Penilaian Otomatis", type="primary"):
            
            st.session_state.all_scores = None 
            st.session_state.submitted = False

            with st.spinner("1/2: Sedang menganalisis gambar dengan Roboflow..."):
                try:
                    # --- Roboflow Detection ---
                    predictions, annotated_image = perform_roboflow_detection(original_image, COMBINED_MODEL_ID)
                    
                    if not predictions:
                        st.warning("Tidak ada elemen UI yang terdeteksi.")
                        st.session_state.annotated_image_path = None
                        st.stop()
                    
                    st.session_state.detection_results = {'predictions': predictions} 

                    st.image(annotated_image, caption="Gambar dengan Deteksi Elemen", use_column_width=True)

                    # Simpan gambar anotasi sementara
                    annotated_image_path = "temp_annotated_image_for_pdf.jpeg"
                    annotated_image.save(annotated_image_path, format='JPEG')
                    st.session_state.annotated_image_path = annotated_image_path
                    st.success(f"Ditemukan **{len(predictions)}** elemen. Lanjut ke penilaian...")

                except Exception as e:
                    st.error(f"Gagal deteksi (Roboflow): {e}")
                    st.session_state.annotated_image_path = None
                    st.session_state.detection_results = None
                    st.stop()

            # --- Gemini Assessment ---
            if st.session_state.detection_results:
                with st.spinner("2/2: Mengirim ke Gemini untuk Penilaian Otomatis..."):
                    predictions_for_gemini = st.session_state.detection_results.get('predictions', [])
                    all_scores_json = get_gemini_assessment(st.session_state.annotated_image_path, predictions_for_gemini)
                    
                    if all_scores_json:
                        st.session_state.all_scores = all_scores_json
                        st.session_state.submitted = True
                        st.session_state.image_name = uploaded_file.name
                        st.success("✅ Penilaian otomatis dari Gemini telah selesai. Unduh laporan di bawah.")
                    else:
                        st.session_state.submitted = False

        
    elif uploaded_file:
        st.info("Tekan tombol 'Mulai Deteksi & Penilaian Otomatis' untuk memproses gambar.")

st.divider()

# ==============================================================================
# 5. UNDUH LAPORAN
# ==============================================================================

st.header("3. Unduh Laporan")

if st.session_state.get('submitted', False) and st.session_state.get('all_scores'):
    
    report_image_path = st.session_state.get('annotated_image_path')
    original_image_pil = st.session_state.get('original_image_pil')
    raw_predictions = st.session_state.detection_results.get('predictions', [])

    if report_image_path and original_image_pil and raw_predictions:
        with st.spinner("Sedang membuat laporan PDF..."):
            
            pdf_output = generate_pdf_report(
                st.session_state.all_scores, 
                report_image_path, 
                original_image_pil, 
                raw_predictions, 
                st.session_state.image_name
            )

        st.download_button(
            label="📥 **Unduh Laporan Penilaian UI (PDF)**",
            data=pdf_output,
            file_name=f"Laporan_Penilaian_UI_{os.path.basename(st.session_state.image_name).replace('.', '_')}_{time.strftime('%Y%m%d%H%M%S')}.pdf",
            mime="application/pdf"
        )
    else:
        st.error("Kesalahan: Data yang diperlukan untuk membuat PDF tidak lengkap.")

    # --- Tampilan JSON Mentah ---
    with st.expander("Lihat Data Penilaian Mentah (Debugging)"):
        st.json(st.session_state.all_scores)