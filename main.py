import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os

# Intento de carga condicional para detección de ventanas (solo Windows)
try:
    import pygetwindow as gw
    HAS_GW = True
except ImportError:
    HAS_GW = False
    print("Advertencia: Librería 'pygetwindow' no encontrada. La detección de cambio de ventana estará desactivada.")

# ==========================================
# CLASE 1: MONITOR DE SISTEMA (NUEVA INTEGRACIÓN)
# ==========================================
class SystemMonitor:
    """
    Encargada exclusivamente de verificar si la ventana del examen es la activa
    en el Sistema Operativo.
    """
    def __init__(self, app_window_title):
        self.app_window_title = app_window_title

    def is_focus_lost(self):
        """Retorna True si el usuario cambió de ventana (Alt-Tab o click fuera)."""
        if not HAS_GW: return False
        
        try:
            window = gw.getActiveWindow()
            if window is None:
                return False 
            
            # Verificamos si el título de la ventana activa coincide con nuestra App
            # Usamos 'in' para ser flexibles con el título
            if self.app_window_title not in window.title and "tk" not in window.title.lower():
                return True
            return False
        except Exception:
            # Si falla la API de windows, no bloqueamos el examen
            return False

# ==========================================
# CLASE 2: PROCESAMIENTO DE OJOS (ROI Y PUPILA)
# ==========================================
class EyeProcessor:
    """
    Lógica de visión por computadora para detectar pupilas usando
    geometría facial y binarización adaptativa.
    """
    def __init__(self):
        self.thresh_val = 40
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 

    def get_eye_roi_geometric(self, face_img):
        """Recorta la zona de los ojos basándose en porcentajes fijos del rostro."""
        h, w = face_img.shape
        # Coordenadas ajustadas para evitar cejas y pómulos
        ly1, ly2 = int(h * 0.30), int(h * 0.52)
        lx1, lx2 = int(w * 0.12), int(w * 0.46)
        rx1, rx2 = int(w * 0.54), int(w * 0.88)

        left_eye_img = face_img[ly1:ly2, lx1:lx2]
        right_eye_img = face_img[ly1:ly2, rx1:rx2]
        
        offsets = ((lx1, ly1), (rx1, ly1))
        return left_eye_img, right_eye_img, offsets

    def find_pupil_blob(self, eye_img):
        """Busca el centroide de la pupila."""
        if eye_img.size == 0: return None
        h_eye, w_eye = eye_img.shape

        blur = cv2.GaussianBlur(eye_img, (5, 5), 0)
        min_val, _, _, _ = cv2.minMaxLoc(blur)
        
        # Umbral dinámico: Pixel más oscuro + tolerancia
        _, binary = cv2.threshold(blur, min_val + 40, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel_open)
        
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not cnts: return None

        best_pupil = None
        best_score = 0

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 6 or area > (h_eye * w_eye) * 0.3: continue
            
            x, y, cw, ch = cv2.boundingRect(c)
            if y <= 1: continue # Descartar pelo en borde superior

            aspect_ratio = cw / float(ch)
            if not (0.6 <= aspect_ratio <= 1.4): continue
            
            # Puntuación: Preferimos blobs centrados verticalmente
            center_dist = abs((y + ch/2) - (h_eye/2))
            score = area - (center_dist * 2)

            if score > best_score:
                best_score = score
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    best_pupil = ((cx / w_eye, cy / h_eye), (cx, cy))

        return best_pupil

# ==========================================
# CLASE 3: TRACKER HÍBRIDO (Lucas-Kanade + Haar)
# ==========================================
class HybridTracker:
    def __init__(self):
        # Configuración para Flujo Óptico (Solem Cap. 10)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.03, minDistance=7, blockSize=7)

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
        self.base_eyes_ratios = [0.5, 0.5] 
        self.calib_eyes = []
        self.calib_head = []

        # Tolerancias
        self.th_head_x = 25
        self.th_head_y = 20
        self.th_eye_sensitivity = 0.07 

    def initialize_tracking(self, gray):
        """Busca rostro inicial para anclar los puntos de seguimiento."""
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
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
        self.p0 = None 

    def process(self, frame):
        """Procesa el frame y retorna imagen anotada + estado detectado."""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()
        status = "Ausente"
        sub_status = ""

        # Tracking no iniciado o perdido
        if self.p0 is None:
            if self.initialize_tracking(frame_gray):
                status = "Inicializando..."
            else:
                cv2.putText(vis, "NO SE DETECTA ROSTRO", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                return vis, "Ausente"

        # Calcular movimiento con Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)

        if p1 is not None and len(p1[st == 1]) > 10:
            good_new = p1[st == 1]
            curr_head_center = np.mean(good_new, axis=0)
            
            # Dibujar puntos
            for pt in good_new:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

            # Estimar rectángulo facial basado en movimiento
            movement = curr_head_center - self.initial_points_center
            ix, iy, iw, ih = self.initial_face_rect
            curr_x, curr_y = int(ix + movement[0]), int(iy + movement[1])
            curr_x, curr_y = max(0, curr_x), max(0, curr_y)
            cv2.rectangle(vis, (curr_x, curr_y), (curr_x + iw, curr_y + ih), (100, 100, 100), 1)

            # Procesar Ojos
            face_roi = frame_gray[curr_y:curr_y+ih, curr_x:curr_x+iw]
            current_ratios = []
            
            if face_roi.size > 0 and iw > 50:
                l_img, r_img, offsets = self.eye_proc.get_eye_roi_geometric(face_roi)
                for idx, (eye_img, offset) in enumerate(zip([l_img, r_img], offsets)):
                    res = self.eye_proc.find_pupil_blob(eye_img)
                    if res:
                        (rx, ry), (cx, cy) = res
                        current_ratios.append(rx) 
                        g_ox, g_oy = curr_x + offset[0], curr_y + offset[1]
                        cv2.rectangle(vis, (g_ox, g_oy), (g_ox + eye_img.shape[1], g_oy + eye_img.shape[0]), (255, 255, 0), 1)
                        cv2.circle(vis, (g_ox + cx, g_oy + cy), 4, (0, 0, 255), -1)
            
            # Lógica de Estado
            if self.calibrating:
                self.calib_head.append(curr_head_center)
                if len(current_ratios) == 2:
                    self.calib_eyes.append(np.mean(current_ratios))
                
                pct = len(self.calib_head)
                cv2.putText(vis, f"CALIBRANDO... {pct}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                
                if pct > 60: 
                    self.base_head_center = np.mean(self.calib_head, axis=0)
                    self.base_eyes_ratios = np.mean(self.calib_eyes) if self.calib_eyes else 0.5
                    self.calibrating = False
                    self.is_calibrated = True

            elif self.is_calibrated:
                # Detección de Postura (Cabeza)
                dx = curr_head_center[0] - self.base_head_center[0]
                dy = curr_head_center[1] - self.base_head_center[1]
                
                head_status = "Centro"
                if abs(dx) > self.th_head_x:
                    head_status = "Izquierda" if dx > 0 else "Derecha" 
                elif abs(dy) > self.th_head_y:
                    head_status = "Abajo" if dy > 0 else "Arriba"

                # Detección de Mirada (Ojos)
                gaze_status = "Centro"
                if head_status == "Centro" and len(current_ratios) > 0:
                    curr_avg_ratio = np.mean(current_ratios)
                    diff = curr_avg_ratio - self.base_eyes_ratios
                    
                    if abs(diff) > self.th_eye_sensitivity:
                        gaze_status = "Ojos_Izquierda" if diff > 0 else "Ojos_Derecha"

                # Prioridades
                if head_status != "Centro":
                    status = head_status
                    sub_status = "(Cabeza)"
                    color = (0, 0, 255)
                elif gaze_status != "Centro":
                    status = gaze_status
                    sub_status = "(Ojos)"
                    color = (0, 165, 255)
                else:
                    status = "Atento"
                    color = (0, 255, 0)

                cv2.putText(vis, f"{status} {sub_status}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                # Caja de referencia segura
                bx, by = int(self.base_head_center[0]), int(self.base_head_center[1])
                cv2.rectangle(vis, (bx - self.th_head_x, by - self.th_head_y), 
                              (bx + self.th_head_x, by + self.th_head_y), (255,255,255), 1)

            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
            
        else:
            self.p0 = None
            status = "Ausente"

        return vis, status

# ==========================================
# CLASE 4: GESTIÓN DE SESIÓN
# ==========================================
class ExamSession:
    def __init__(self):
        self.active = False
        self.stats = {}
        self.last_time = 0
    
    def start(self):
        self.active = True
        self.last_time = time.time()
        # Inicializamos contadores incluyendo 'Ventana'
        self.stats = {
            "total": 0, "ok": 0, "bad": 0, 
            "detail": {
                "Izquierda":0, "Derecha":0, "Arriba":0, "Abajo":0,
                "Ojos_Izquierda":0, "Ojos_Derecha":0, "Ventana":0, "Ausente":0
            }
        }

    def stop(self): 
        self.active = False

    def update(self, st):
        if not self.active: return
        
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        self.stats["total"] += dt
        
        # 'Ventana' cuenta como distracción
        is_attentive = "Atento" in st or "Centro" in st or "Inicializando" in st
        
        if is_attentive:
            self.stats["ok"] += dt
        else:
            self.stats["bad"] += dt
            key = st.split(" ")[0] 
            if key in self.stats["detail"]:
                self.stats["detail"][key] += dt
            else:
                self.stats["detail"]["Ausente"] += dt

    def get_suspicion_percentage(self):
        if self.stats["total"] < 1: return 0
        return (self.stats["bad"] / self.stats["total"]) * 100

# ==========================================
# APP PRINCIPAL (TKINTER)
# ==========================================
class ProctorApp:
    def __init__(self, root):
        self.root = root
        self.title_window = "Sistema de Monitoreo de Examen"
        self.root.title(self.title_window)
        
        self.tracker = HybridTracker()
        self.session = ExamSession()
        # Instanciamos el monitor de ventanas
        self.sys_monitor = SystemMonitor(self.title_window)

        tk.Label(root, text="MONITOR DE EXAMEN - VISIÓN ARTIFICIAL", font=("Arial", 14, "bold")).pack(pady=10)
        
        self.canvas = tk.Canvas(root, width=640, height=480, bg="#333")
        self.canvas.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=15)
        
        self.btn_calib = tk.Button(btn_frame, text="1. CALIBRAR", command=self.do_calib, 
                                   bg="#555", fg="white", font=("Arial", 10), width=15)
        self.btn_calib.pack(side=tk.LEFT, padx=10)
        
        self.btn_start = tk.Button(btn_frame, text="2. INICIAR EXAMEN", command=self.start_exam, 
                                   state=tk.DISABLED, bg="#28a745", fg="white", font=("Arial", 10, "bold"), width=15)
        self.btn_start.pack(side=tk.LEFT, padx=10)
        
        self.btn_stop = tk.Button(btn_frame, text="3. FINALIZAR", command=self.stop_exam, 
                                  state=tk.DISABLED, bg="#dc3545", fg="white", font=("Arial", 10, "bold"), width=15)
        self.btn_stop.pack(side=tk.LEFT, padx=10)

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
        suspicion = self.session.get_suspicion_percentage()
        
        # Formato de reporte solicitado por el usuario (Estilo Código 2)
        status_final = "REPROBADO" if suspicion > 40 else "APROBADO"
        
        report = f"Resultados:\n\n"
        report += f"Sospecha Total: {suspicion:.2f}%\n"
        report += f"Estado: {status_final}\n\n"
        report += "Detalle (segundos):\n"
        
        for k, v in self.session.stats['detail'].items():
            if v > 0.5: 
                report += f"- {k}: {v:.1f}s\n"
            
        messagebox.showinfo("Reporte Final", report)
        self.cleanup()

    def cleanup(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.quit()

    def loop(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # Lógica Prioritaria: Chequeo de Ventana Activa
            # Si el examen está activo y se pierde el foco, sobrescribimos todo.
            window_lost = False
            if self.session.active:
                if self.sys_monitor.is_focus_lost():
                    window_lost = True
                    status = "Ventana"
                    # Overlay visual de Alerta (Estilo Código 2)
                    cv2.putText(frame, "ALERTA: VENTANA INACTIVA", (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    
                    # Dibujamos rectángulo rojo alrededor de toda la imagen
                    h, w, _ = frame.shape
                    cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 10)
            
            # Si no se perdió la ventana, ejecutamos tracking normal
            if not window_lost:
                vis, status = self.tracker.process(frame)
            else:
                vis = frame # Usamos el frame con la alerta roja

            self.session.update(status)
            
            if self.tracker.is_calibrated and not self.session.active:
                self.btn_start.config(state=tk.NORMAL)

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