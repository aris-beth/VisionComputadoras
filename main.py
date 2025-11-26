import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os

# Intento de importar librería de ventanas
try:
    import pygetwindow as gw
    HAS_GW = True
except ImportError:
    HAS_GW = False
    print("Nota: pygetwindow no instalado.")

# ==========================================
# CLASE 1: PROCESAMIENTO DE OJOS (OPTIMIZADO - GEOMÉTRICO)
# ==========================================
class EyeProcessor:
    def __init__(self):
        self.thresh_val = 40
        # Kernel elíptico funciona mejor para ojos que cuadrado
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 

    def get_eye_roi_geometric(self, face_img):
        h, w = face_img.shape
        # CAMBIO 1: Bajamos el inicio de Y del 25% al 30% para evitar cejas/fleco
        ly1, ly2 = int(h * 0.30), int(h * 0.52) # Antes era 0.25 - 0.50
        
        # Ajustamos un poco X para centrarnos más
        lx1, lx2 = int(w * 0.12), int(w * 0.46)
        rx1, rx2 = int(w * 0.54), int(w * 0.88)

        left_eye_img = face_img[ly1:ly2, lx1:lx2]
        right_eye_img = face_img[ly1:ly2, rx1:rx2]
        
        offsets = ((lx1, ly1), (rx1, ly1))
        return left_eye_img, right_eye_img, offsets

    def find_pupil_blob(self, eye_img):
        """
        Busca la pupila filtrando por forma y posición para evitar cabello.
        """
        if eye_img.size == 0: return None
        h_eye, w_eye = eye_img.shape

        # 1. Preprocesamiento
        blur = cv2.GaussianBlur(eye_img, (5, 5), 0)
        
        # 2. Umbralización Adaptativa (mismo método que antes)
        min_val, _, _, _ = cv2.minMaxLoc(blur)
        thresh_limit = min_val + 40 # Subí un poco la tolerancia
        _, binary = cv2.threshold(blur, thresh_limit, 255, cv2.THRESH_BINARY_INV)

        # 3. Limpieza
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel_open)
        
        # 4. Análisis de Contornos (MEJORADO)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not cnts: return None

        best_pupil = None
        best_score = 0

        for c in cnts:
            area = cv2.contourArea(c)
            # Filtro de tamaño mínimo/máximo
            if area < 6 or area > (h_eye * w_eye) * 0.3:
                continue
            
            x, y, cw, ch = cv2.boundingRect(c)
            
            # CAMBIO 2: FILTRO DE CABELLO (Bordes)
            # Si el contorno toca el borde superior (y=0), es cabello entrando -> IGNORAR
            if y <= 1: 
                continue

            # CAMBIO 3: FILTRO DE FORMA (Aspect Ratio)
            # La pupila cabe en un cuadrado. El ratio (ancho/alto) debe ser cercano a 1.
            aspect_ratio = cw / float(ch)
            # Permitimos entre 0.6 y 1.4 (un poco ovalado está bien, pero no una línea)
            if not (0.6 <= aspect_ratio <= 1.4):
                continue
            
            # Puntuamos: Preferimos el que esté más cerca del centro vertical y sea más grande
            center_dist = abs((y + ch/2) - (h_eye/2))
            score = area - (center_dist * 2) # Penalizar si está muy arriba o muy abajo

            if score > best_score:
                best_score = score
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    best_pupil = ((cx / w_eye, cy / h_eye), (cx, cy))

        return best_pupil

# ==========================================
# CLASE 2: TRACKER HÍBRIDO (MODIFICADO)
# ==========================================
class HybridTracker:
    def __init__(self):
        # Params Optical Flow (Solem Cap 10)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.03, minDistance=7, blockSize=7)

        # Detector Facial
        path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(os.path.join(path, 'haarcascade_frontalface_default.xml'))
        self.eye_proc = EyeProcessor()

        # Variables de estado
        self.old_gray = None
        self.p0 = None
        self.initial_face_rect = None 
        self.initial_points_center = None

        # Calibración
        self.calibrating = False
        self.is_calibrated = False
        self.base_head_center = None
        self.base_eyes_ratios = [0.5, 0.5] # [Left_X, Right_X] promedio

        # Buffers
        self.calib_eyes = []
        self.calib_head = []

        # Umbrales
        self.th_head_x = 25
        self.th_head_y = 20
        # Sensibilidad de ojos (menor número = más sensible)
        self.th_eye_sensitivity = 0.07 

    def initialize_tracking(self, gray):
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            # Tomar la cara más grande
            (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
            
            # Crear máscara para buscar puntos SOLO en la cara
            mask = np.zeros_like(gray)
            mask[y:y+h, x:x+w] = 255
            
            p = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            if p is not None:
                self.p0 = p
                self.old_gray = gray.copy()
                self.initial_face_rect = (x, y, w, h)
                self.initial_points_center = np.mean(p, axis=0)[0]
                return True
        return False

    def start_calib(self):
        self.calibrating = True
        self.is_calibrated = False
        self.calib_eyes = []
        self.calib_head = []
        self.p0 = None # Reiniciar tracker

    def process(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()
        status = "Ausente"
        sub_status = ""

        # 1. Gestión del Tracking (Optical Flow)
        if self.p0 is None:
            if self.initialize_tracking(frame_gray):
                status = "Inicializando..."
            else:
                cv2.putText(vis, "NO SE DETECTA ROSTRO", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                return vis, "Ausente"

        # Calcular Flujo Óptico
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)

        # Verificar calidad del tracking
        if p1 is not None and len(p1[st == 1]) > 10:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            
            # Centro actual de los puntos trackeados
            curr_head_center = np.mean(good_new, axis=0)
            
            # Dibujar puntos de tracking
            for pt in good_new:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

            # Estimar nueva posición del rectángulo de la cara
            # Movimiento relativo desde el inicio
            movement = curr_head_center - self.initial_points_center
            ix, iy, iw, ih = self.initial_face_rect
            curr_x, curr_y = int(ix + movement[0]), int(iy + movement[1])
            
            # Validar límites
            curr_x = max(0, curr_x)
            curr_y = max(0, curr_y)
            
            # Dibujar cara
            cv2.rectangle(vis, (curr_x, curr_y), (curr_x + iw, curr_y + ih), (100, 100, 100), 1)

            # 2. Procesamiento de Ojos (Usando ROI Geométrico)
            face_roi = frame_gray[curr_y:curr_y+ih, curr_x:curr_x+iw]
            
            # Validar que el ROI sea válido
            current_ratios = []
            if face_roi.size > 0 and iw > 50:
                l_img, r_img, offsets = self.eye_proc.get_eye_roi_geometric(face_roi)
                
                # Procesar ambos ojos
                for idx, (eye_img, offset) in enumerate(zip([l_img, r_img], offsets)):
                    res = self.eye_proc.find_pupil_blob(eye_img)
                    if res:
                        (rx, ry), (cx, cy) = res
                        current_ratios.append(rx) # Guardamos solo X para simplificar izquierda/derecha
                        
                        # Dibujar Ojo y Pupila
                        g_ox = curr_x + offset[0]
                        g_oy = curr_y + offset[1]
                        cv2.rectangle(vis, (g_ox, g_oy), (g_ox + eye_img.shape[1], g_oy + eye_img.shape[0]), (255, 255, 0), 1)
                        cv2.circle(vis, (g_ox + cx, g_oy + cy), 4, (0, 0, 255), -1)
            
            # 3. Lógica de Estado y Calibración
            if self.calibrating:
                self.calib_head.append(curr_head_center)
                if len(current_ratios) == 2: # Solo si detectamos ambos ojos
                    self.calib_eyes.append(np.mean(current_ratios))
                
                pct = len(self.calib_head)
                cv2.putText(vis, f"CALIBRANDO... {pct}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                
                if pct > 60: # Fin calibración
                    self.base_head_center = np.mean(self.calib_head, axis=0)
                    if self.calib_eyes:
                        self.base_eyes_ratios = np.mean(self.calib_eyes)
                    else:
                        self.base_eyes_ratios = 0.5
                    
                    self.calibrating = False
                    self.is_calibrated = True
                    print(f"Calibrado: Head {self.base_head_center}, EyeRatio {self.base_eyes_ratios:.2f}")

            elif self.is_calibrated:
                # A. Chequeo Cabeza (Head Pose por desplazamiento de puntos)
                dx = curr_head_center[0] - self.base_head_center[0]
                dy = curr_head_center[1] - self.base_head_center[1]
                
                head_status = "Centro"
                if abs(dx) > self.th_head_x:
                    head_status = "Izquierda" if dx > 0 else "Derecha" # Espejo
                elif abs(dy) > self.th_head_y:
                    head_status = "Abajo" if dy > 0 else "Arriba"

                # B. Chequeo Ojos (Gaze)
                gaze_status = "Centro"
                if head_status == "Centro" and len(current_ratios) > 0:
                    curr_avg_ratio = np.mean(current_ratios)
                    diff = curr_avg_ratio - self.base_eyes_ratios
                    
                    # Debug en pantalla
                    cv2.putText(vis, f"EyeDiff: {diff:.3f}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    
                    if abs(diff) > self.th_eye_sensitivity:
                        # Nota: La dirección depende de cómo detecta la cámara (espejo)
                        # Si ratio aumenta -> pupila va a la derecha (del cuadro) -> usuario mira izquierda
                        gaze_status = "Ojos_Izquierda" if diff > 0 else "Ojos_Derecha"

                # Decisión Final
                if head_status != "Centro":
                    status = head_status
                    sub_status = "(Cabeza)"
                    color = (0, 0, 255)
                elif gaze_status != "Centro":
                    status = gaze_status
                    sub_status = "(Ojos)"
                    color = (0, 165, 255) # Naranja
                else:
                    status = "Atento"
                    color = (0, 255, 0)

                # UI Final
                cv2.putText(vis, f"{status} {sub_status}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                # Dibujar caja de referencia central
                bx, by = int(self.base_head_center[0]), int(self.base_head_center[1])
                cv2.rectangle(vis, (bx - self.th_head_x, by - self.th_head_y), 
                              (bx + self.th_head_x, by + self.th_head_y), (255,255,255), 1)

            # Actualizar optical flow
            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
            
        else:
            # Perdimos el tracking
            self.p0 = None
            status = "Ausente"

        return vis, status

# ==========================================
# APP PRINCIPAL (Mantiene tu lógica de Stats)
# ==========================================
class ExamSession:
    def __init__(self):
        self.active = False
        self.stats = {"total": 0, "ok": 0, "bad": 0, "detail": {}}
        self.last = 0
    
    def start(self):
        self.active = True
        self.last = time.time()
        self.stats = {"total": 0, "ok": 0, "bad": 0, 
                      "detail": {"Izquierda":0, "Derecha":0, "Arriba":0, "Abajo":0,
                                 "Ojos_Izquierda":0, "Ojos_Derecha":0, "Ventana":0, "Ausente":0}}

    def stop(self): self.active = False

    def update(self, st):
        if not self.active: return
        now = time.time()
        dt = now - self.last
        self.last = now
        self.stats["total"] += dt
        
        # Clasificar atención
        if "Atento" in st or "Centro" in st or "Inicializando" in st:
            self.stats["ok"] += dt
        else:
            self.stats["bad"] += dt
            # Limpiar string para la llave del dict (quitar parentesis extra)
            key = st.split(" ")[0] 
            if key in self.stats["detail"]:
                self.stats["detail"][key] += dt
            else:
                self.stats["detail"]["Ausente"] += dt

    def get_suspicion(self):
        if self.stats["total"] < 1: return 0
        return (self.stats["bad"] / self.stats["total"]) * 100

class ProctorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Monitor Examen - Vision Clásica")
        self.tracker = HybridTracker()
        self.session = ExamSession()

        tk.Label(root, text="SISTEMA DE MONITOREO - EXAMEN", font=("Arial", 14, "bold")).pack(pady=5)
        self.canvas = tk.Canvas(root, width=640, height=480, bg="#222")
        self.canvas.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        
        self.btn_calib = tk.Button(btn_frame, text="1. CALIBRAR", command=self.do_calib, bg="#444", fg="white", width=15)
        self.btn_calib.pack(side=tk.LEFT, padx=5)
        
        self.btn_start = tk.Button(btn_frame, text="2. INICIAR EXAMEN", command=self.start_exam, state=tk.DISABLED, bg="green", fg="white", width=15)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = tk.Button(btn_frame, text="3. FINALIZAR", command=self.stop_exam, state=tk.DISABLED, bg="red", fg="white", width=15)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.cap = cv2.VideoCapture(0)
        self.loop()

    def do_calib(self):
        self.tracker.start_calib()
        
    def start_exam(self):
        self.session.start()
        self.btn_calib.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        
    def stop_exam(self):
        self.session.stop()
        suspicion = self.session.get_suspicion()
        res = "APROBADO" if suspicion <= 40 else "REVISIÓN REQUERIDA"
        
        report = f"Resultado: {res}\n"
        report += f"Tiempo Distraído: {suspicion:.1f}%\n\n"
        report += "Detalles (segundos):\n"
        for k, v in self.session.stats['detail'].items():
            if v > 0.5: report += f"- {k}: {v:.1f}s\n"
            
        messagebox.showinfo("Reporte Final", report)
        self.root.quit()

    def loop(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            vis, status = self.tracker.process(frame)

            # Detectar ventana activa (Windows)
            if self.session.active and HAS_GW:
                try:
                    win = gw.getActiveWindow()
                    # Modifica "Monitor" o "python" según el título de tu ventana al ejecutar
                    if win and "Monitor" not in win.title and "tk" not in win.title.lower():
                        status = "Ventana"
                        cv2.putText(vis, "ALERTA: OTRA VENTANA", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                except: pass

            self.session.update(status)
            
            # Habilitar botón de inicio post-calibración
            if self.tracker.is_calibrated and not self.session.active:
                self.btn_start.config(state=tk.NORMAL)

            # Mostrar video en Tkinter
            img = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk
        
        self.root.after(20, self.loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = ProctorApp(root)
    root.mainloop()