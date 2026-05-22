import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk

# Mengatur tema dasar ke Light Mode agar warna cerah berfungsi maksimal
ctk.set_appearance_mode("light")

# Konstanta Warna Tema Cream & Cokelat Estetik (Hex Color)
WARNA_BG_UTAMA = "#FDFBF7"       # Cream sangat terang untuk background dasar
WARNA_CARD = "#F5EFE6"           # Cream medium untuk latar belakang panel/box
WARNA_TEKS_UTAMA = "#4A3F35"     # Cokelat tua gelap untuk tulisan agar kontras dan terbaca
WARNA_TEKS_INFO = "#A09383"       # Cokelat muda untuk teks petunjuk
WARNA_TOMBOL_PILIH = "#E8D5C4"   # Soft cream-brown untuk tombol pilih file
WARNA_HOVER_PILIH = "#D8B4A0"
WARNA_TOMBOL_SCAN = "#8FA89B"    # Hijau sage lembut untuk tombol aksi scan (agar tetap terlihat menonjol)
WARNA_HOVER_SCAN = "#768F82"

# ==========================================
# 1. FUNGSI PENGOLAHAN CITRA (CORE LOGIC)
# ==========================================

def order_points(pts):
    """
    Mengurutkan 4 titik sudut agar konsisten:
    [top-left, top-right, bottom-right, bottom-left]
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # top-left akan memiliki jumlah (x + y) terkecil
    # bottom-right akan memiliki jumlah (x + y) terbesar
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # top-right akan memiliki selisih (y - x) terkecil / (x - y) terbesar
    # bottom-left akan memiliki selisih (y - x) terbesar / (x - y) terkecil
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def process_scan(image_path):
    """
    Memproses gambar untuk mendeteksi kontur kertas, meluruskan, dan memberikan efek scan.
    """
    orig_img = cv2.imread(image_path)
    img_display = orig_img.copy()
    
    # Pre-processing
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    
    # Mencari Kontur Dokumen
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    doc_contour = None
    for c in contours:
        # Perkirakan keliling kontur
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # Jika kontur memiliki 4 titik sudut, kita asumsikan itu adalah dokumen
        if len(approx) == 4:
            doc_contour = approx
            break
            
    # Jika tidak ditemukan kontur 4 sudut, gunakan sudut gambar
    if doc_contour is None:
        h, w = orig_img.shape[:2]
        doc_contour = np.array([[[0,0]], [[w,0]], [[w,h]], [[0,h]]])

    # Mengambil dan mengurutkan 4 titik sudut
    pts = doc_contour.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Menggambar 4 titik merah (Anchor Points) pada Input View
    for point in rect:
        cv2.circle(img_display, (int(point[0]), int(point[1])), int(orig_img.shape[0]*0.015), (0, 0, 255), -1)
        
    # Warp Perspective (Meluruskan Gambar)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig_img, M, (max_width, max_height))
    
    # Final Touch (Adaptive Thresholding untuk efek scan bersih)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    scanned_effect = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 10)
    
    return img_display, scanned_effect

# ==========================================
# 2. ELEMEN USER INTERFACE MODERN & MENARIK (KUSTOM CREAM)
# ==========================================

class AestheticScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("UTS PCD - Document Scanner Aesthetic Pro")
        self.root.geometry("980x700")
        self.root.configure(fg_color=WARNA_BG_UTAMA) # Background jendela utama
        self.file_path = None
        
        # Header Aplikasi
        self.title_label = ctk.CTkLabel(root, text="DOCUMENT SCANNER PRO", text_color=WARNA_TEKS_UTAMA, font=ctk.CTkFont(family="Helvetica", size=28, weight="bold"))
        self.title_label.pack(pady=20)
        
        # Frame Utama untuk menampung Kontrol dan Gambar
        self.main_frame = ctk.CTkFrame(root, fg_color=WARNA_BG_UTAMA)
        self.main_frame.pack(fill="both", expand=True, padx=25, pady=15)
        
        # Panel Kontrol (Tombol-tombol)
        self.control_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.control_frame.pack(fill="x", pady=15)
        
        # Ikon Dokumen untuk Tombol Pilih (Pastikan file ikon ada di folder yang sama)
        try:
            self.doc_icon = ctk.CTkImage(light_image=Image.open("doc_icon.png"), dark_image=Image.open("doc_icon.png"), size=(20, 20))
            self.btn_select = ctk.CTkButton(self.control_frame, text="📁 Pilih Foto Dokumen", font=ctk.CTkFont(size=14, weight="bold"), image=self.doc_icon, fg_color=WARNA_TOMBOL_PILIH, hover_color=WARNA_HOVER_PILIH, text_color=WARNA_TEKS_UTAMA, command=self.open_image, height=45, corner_radius=10)
        except:
            self.btn_select = ctk.CTkButton(self.control_frame, text="📁 Pilih Foto Dokumen", font=ctk.CTkFont(size=14, weight="bold"), fg_color=WARNA_TOMBOL_PILIH, hover_color=WARNA_HOVER_PILIH, text_color=WARNA_TEKS_UTAMA, command=self.open_image, height=45, corner_radius=10)
        self.btn_select.pack(side="left", padx=25, expand=True, fill="x")
        
        # Ikon Scan untuk Tombol Scan
        try:
            self.scan_icon = ctk.CTkImage(light_image=Image.open("scan_icon.png"), dark_image=Image.open("scan_icon.png"), size=(20, 20))
            self.btn_scan = ctk.CTkButton(self.control_frame, text="⚡ Scan / Warp Document", font=ctk.CTkFont(size=14, weight="bold"), image=self.scan_icon, fg_color=WARNA_TOMBOL_SCAN, hover_color=WARNA_HOVER_SCAN, text_color="white", command=self.scan_image, state="disabled", height=45, corner_radius=10)
        except:
            self.btn_scan = ctk.CTkButton(self.control_frame, text="⚡ Scan / Warp Document", font=ctk.CTkFont(size=14, weight="bold"), fg_color=WARNA_TOMBOL_SCAN, hover_color=WARNA_HOVER_SCAN, text_color="white", command=self.scan_image, state="disabled", height=45, corner_radius=10)
        self.btn_scan.pack(side="right", padx=25, expand=True, fill="x")
        
        # Tampilan Gambar (Grid Kiri dan Kanan)
        self.view_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.view_frame.pack(fill="both", expand=True, padx=15, pady=20)
        
        # Sisi Kiri (Input) dengan Bayangan
        self.left_panel_outer = ctk.CTkFrame(self.view_frame, fg_color=WARNA_CARD, corner_radius=15, border_width=1, border_color="#E0D8D0") # Sedikit shadow & border
        self.left_panel_outer.pack(side="left", fill="both", expand=True, padx=15)
        
        self.left_panel_inner = ctk.CTkFrame(self.left_panel_outer, fg_color="transparent")
        self.left_panel_inner.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.lbl_input = ctk.CTkLabel(self.left_panel_inner, text="INPUT VIEW (Original + Anchors)", text_color=WARNA_TEKS_UTAMA, font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_input.pack(pady=10)
        
        self.canvas_input = ctk.CTkLabel(self.left_panel_inner, text="Belum ada gambar yang dipilih", text_color=WARNA_TEKS_INFO, font=ctk.CTkFont(size=11, slant="italic"))
        self.canvas_input.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Sisi Kanan (Output) dengan Bayangan
        self.right_panel_outer = ctk.CTkFrame(self.view_frame, fg_color=WARNA_CARD, corner_radius=15, border_width=1, border_color="#E0D8D0") # Sedikit shadow & border
        self.right_panel_outer.pack(side="right", fill="both", expand=True, padx=15)
        
        self.right_panel_inner = ctk.CTkFrame(self.right_panel_outer, fg_color="transparent")
        self.right_panel_inner.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.lbl_output = ctk.CTkLabel(self.right_panel_inner, text="OUTPUT VIEW (Scanned Result)", text_color=WARNA_TEKS_UTAMA, font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_output.pack(pady=10)
        
        self.canvas_output = ctk.CTkLabel(self.right_panel_inner, text="Menunggu proses scan...", text_color=WARNA_TEKS_INFO, font=ctk.CTkFont(size=11, slant="italic"))
        self.canvas_output.pack(fill="both", expand=True, padx=20, pady=20)

    def open_image(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if self.file_path:
            img = cv2.imread(self.file_path)
            self.display_image(img, self.canvas_input)
            self.btn_scan.configure(state="normal")
            self.canvas_output.configure(text="Menunggu proses scan...", font=ctk.CTkFont(size=11, slant="italic"))

    def scan_image(self):
        if self.file_path:
            input_with_anchors, final_output = process_scan(self.file_path)
            self.display_image(input_with_anchors, self.canvas_input)
            self.display_image(final_output, self.canvas_output, is_gray=True)

    def display_image(self, img, label_widget, is_gray=False):
        h, w = img.shape[:2]
        max_size = 380
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        
        if is_gray:
            img_rgb = Image.fromarray(img_resized)
        else:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_rgb = Image.fromarray(img_rgb)
            
        img_tk = ctk.CTkImage(light_image=img_rgb, dark_image=img_rgb, size=(new_w, new_h))
        label_widget.configure(image=img_tk, text="")
        label_widget.image = img_tk

if __name__ == "__main__":
    # Menghapus Window standar Tkinter dan menggunakan CustomTkinter
    root = ctk.CTk()
    app = AestheticScannerApp(root)
    root.mainloop()