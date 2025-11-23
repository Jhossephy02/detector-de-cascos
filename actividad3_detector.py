# actividad3_detector.py
"""
ACTIVIDAD 3 - Sistema de Detección de Personas
Detecta personas, reproduce audio y envía alertas por WhatsApp/SMS

INSTRUCCIONES DE USO:
1. Asegúrate de tener instaladas las dependencias: pip install -r requirements.txt
2. Configura .env con tus credenciales de Twilio (opcional)
3. Ejecuta: python actividad3_detector.py
4. El sistema detectará personas y enviará alertas automáticamente

CONTROLES:
- Q/ESC: Salir
- S: Guardar screenshot
- R: Resetear cooldown de alertas
"""

import cv2
import numpy as np
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Importar configuración
try:
    from config import *
except ImportError:
    print("❌ Error: config.py no encontrado")
    sys.exit(1)


class PersonDetectionSystem:
    """Sistema de detección de personas con alertas"""
    
    def __init__(self):
        print("=" * 70)
        print("🎯 ACTIVIDAD 3 - DETECCIÓN DE PERSONAS".center(70))
        print("=" * 70)
        
        # Inicializar componentes
        self.audio = self._init_audio()
        self.twilio = self._init_twilio()
        self.face_cascade = self._init_face_detector()
        
        # Estado de alertas
        self.last_alert_time = None
        self.alert_cooldown_seconds = ALERT_COOLDOWN  # 30 segundos por defecto
        self.total_detections = 0
        self.alerts_sent = 0
        
        # FPS
        self.fps = 0
        self.fps_time = time.time()
        self.fps_count = 0
        
        print("\n✅ Sistema inicializado")
        print(f"   Audio: {'✅' if self.audio else '❌'}")
        print(f"   WhatsApp/SMS: {'✅' if self.twilio else '❌'}")
        print(f"   Detector rostros: {'✅' if self.face_cascade else '❌'}")
        print()
    
    def _init_audio(self):
        """Inicializa el sistema de audio"""
        if not ENABLE_AUDIO:
            return None
        
        try:
            import pygame
            pygame.mixer.init()
            
            # Crear audio si no existe
            if not ALERT_SOUND_PATH.exists():
                self._create_alert_sound()
            
            sound = pygame.mixer.Sound(str(ALERT_SOUND_PATH))
            sound.set_volume(ALERT_VOLUME)
            return sound
        except Exception as e:
            print(f"⚠️  Audio no disponible: {e}")
            return None
    
    def _create_alert_sound(self):
        """Crea el archivo de sonido de alerta"""
        try:
            from scipy.io import wavfile
            sr = 44100
            t = np.linspace(0, ALERT_DURATION, int(sr * ALERT_DURATION))
            
            # Tono de alerta (880 Hz con decay)
            audio = np.sin(2 * np.pi * 880 * t) * np.exp(-3 * t) * 0.3
            audio = (audio * 32767).astype(np.int16)
            
            wavfile.write(str(ALERT_SOUND_PATH), sr, audio)
            print("✅ Archivo de audio creado")
        except Exception as e:
            print(f"⚠️  No se pudo crear audio: {e}")
    
    def _init_twilio(self):
        """Inicializa Twilio para WhatsApp/SMS"""
        if not ENABLE_WHATSAPP:
            return None
        
        try:
            from twilio.rest import Client
            
            if not TWILIO_ACCOUNT_SID or 'tu_' in TWILIO_ACCOUNT_SID:
                print("⚠️  Twilio no configurado (edita .env)")
                return None
            
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            # Verificar credenciales
            client.api.accounts(TWILIO_ACCOUNT_SID).fetch()
            return client
        except Exception as e:
            print(f"⚠️  Twilio no disponible: {e}")
            return None
    
    def _init_face_detector(self):
        """Inicializa el detector de rostros Haar Cascade"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            
            if cascade.empty():
                print("⚠️  Haar Cascade no cargado correctamente")
                return None
            
            return cascade
        except Exception as e:
            print(f"⚠️  Detector de rostros no disponible: {e}")
            return None
    
    def detect_persons(self, frame):
        """Detecta personas (rostros) en el frame"""
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def should_send_alert(self):
        """Verifica si se debe enviar una alerta (cooldown)"""
        if self.last_alert_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_alert_time).total_seconds()
        return elapsed >= self.alert_cooldown_seconds
    
    def play_alert_sound(self):
        """Reproduce el sonido de alerta"""
        if self.audio:
            try:
                self.audio.play()
                print("🔊 Alerta reproducida")
            except Exception as e:
                print(f"❌ Error reproduciendo audio: {e}")
    
    def send_alert_message(self):
        """Envía mensaje por WhatsApp o SMS"""
        if not self.twilio:
            print("⚠️  WhatsApp/SMS no disponible")
            return False
        
        try:
            now = datetime.now()
            mensaje = f"""
🚨 ALERTA DE SEGURIDAD

👤 PERSONA DETECTADA en el área

📅 Fecha: {now.strftime("%d/%m/%Y")}
🕐 Hora: {now.strftime("%H:%M:%S")}
📍 Ubicación: {DEFAULT_LOCATION}

⚡ Sistema de Vigilancia Activo

---
SENATI - Actividad 3
            """.strip()
            
            # Enviar mensaje
            message = self.twilio.messages.create(
                body=mensaje,
                from_=f'whatsapp:{TWILIO_PHONE_NUMBER}',
                to=f'whatsapp:{DESTINATION_PHONE}'
            )
            
            self.last_alert_time = datetime.now()
            self.alerts_sent += 1
            print(f"✅ Mensaje enviado: {message.sid}")
            return True
            
        except Exception as e:
            print(f"❌ Error enviando mensaje: {e}")
            return False
    
    def trigger_alert(self):
        """Activa todas las alertas"""
        print("\n🚨 ¡PERSONA DETECTADA!")
        
        # Reproducir audio
        self.play_alert_sound()
        
        # Enviar mensaje si corresponde
        if self.should_send_alert():
            self.send_alert_message()
        else:
            remaining = self.alert_cooldown_seconds - (
                datetime.now() - self.last_alert_time
            ).total_seconds()
            print(f"⏳ Cooldown activo: {remaining:.0f}s restantes")
    
    def draw_detections(self, frame, faces):
        """Dibuja las detecciones en el frame"""
        for (x, y, w, h) in faces:
            # Rectángulo alrededor del rostro
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Label
            cv2.putText(frame, "PERSONA", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def draw_info_panel(self, frame, num_persons):
        """Dibuja el panel de información"""
        h, w = frame.shape[:2]
        
        # Panel superior negro semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Información
        cv2.putText(frame, "ACTIVIDAD 3 - DETECCION DE PERSONAS", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        status = f"Personas detectadas: {num_persons}"
        color = (0, 255, 0) if num_persons > 0 else (128, 128, 128)
        cv2.putText(frame, status, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.putText(frame, f"Alertas enviadas: {self.alerts_sent}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS en la esquina superior derecha
        cv2.putText(frame, f"FPS: {self.fps:.0f}", (w-120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Cooldown si está activo
        if self.last_alert_time:
            remaining = self.alert_cooldown_seconds - (
                datetime.now() - self.last_alert_time
            ).total_seconds()
            
            if remaining > 0:
                cv2.putText(frame, f"Cooldown: {remaining:.0f}s", (w-150, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
    
    def calculate_fps(self):
        """Calcula los FPS"""
        self.fps_count += 1
        elapsed = time.time() - self.fps_time
        
        if elapsed >= 1.0:
            self.fps = self.fps_count / elapsed
            self.fps_count = 0
            self.fps_time = time.time()
    
    def process_frame(self, frame):
        """Procesa un frame de video"""
        # Detectar personas
        faces = self.detect_persons(frame)
        num_persons = len(faces)
        
        # Si hay personas detectadas
        if num_persons > 0:
            self.total_detections += 1
            self.trigger_alert()
            self.draw_detections(frame, faces)
        
        # Calcular FPS
        self.calculate_fps()
        
        # Dibujar información
        self.draw_info_panel(frame, num_persons)
        
        return frame
    
    def run(self):
        """Ejecuta el sistema de detección"""
        print("\n" + "=" * 70)
        print("🎥 Iniciando cámara...".center(70))
        print("=" * 70 + "\n")
        
        # Abrir cámara
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not cap.isOpened():
            print(f"❌ No se pudo abrir la cámara {CAMERA_INDEX}")
            print("   Intenta cambiar CAMERA_INDEX en .env")
            return
        
        # Configurar cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ Cámara iniciada: {actual_w}x{actual_h}")
        print("\n🎮 CONTROLES:")
        print("   Q/ESC - Salir")
        print("   S     - Guardar screenshot")
        print("   R     - Resetear cooldown")
        print("\n⚡ Sistema activo - Detectando personas...\n")
        print("=" * 70 + "\n")
        
        window_name = 'Actividad 3 - Deteccion de Personas (Q para salir)'
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error capturando frame")
                    break
                
                # Procesar frame
                processed = self.process_frame(frame)
                
                # Mostrar
                cv2.imshow(window_name, processed)
                
                # Controles
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q') or key == 27:
                    print("\n👋 Saliendo...")
                    break
                elif key == ord('s') or key == ord('S'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"captura_{timestamp}.jpg"
                    cv2.imwrite(filename, processed)
                    print(f"📸 Screenshot guardado: {filename}")
                elif key == ord('r') or key == ord('R'):
                    self.last_alert_time = None
                    print("✅ Cooldown reseteado")
                
                # Verificar si se cerró la ventana
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrumpido por el usuario")
        
        finally:
            # Estadísticas finales
            total_time = time.time() - start_time
            
            print("\n" + "=" * 70)
            print("📊 ESTADÍSTICAS FINALES".center(70))
            print("=" * 70)
            print(f"   ⏱️  Tiempo total: {total_time:.1f} segundos")
            print(f"   👤 Detecciones: {self.total_detections}")
            print(f"   📱 Alertas enviadas: {self.alerts_sent}")
            print(f"   🎯 FPS promedio: {self.total_detections/max(total_time, 1):.1f}")
            print("=" * 70 + "\n")
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Forzar cierre en Windows
            for _ in range(5):
                cv2.waitKey(1)


def main():
    """Función principal"""
    print("\n" + "=" * 70)
    print("🎯 ACTIVIDAD 3 - DETECCIÓN DE PERSONAS".center(70))
    print("=" * 70)
    print("\nEste sistema detectará personas y enviará alertas.")
    print("\n📝 Requisitos cumplidos:")
    print("   ✅ Detectar personas")
    print("   ✅ Reproducir alerta de audio")
    print("   ✅ Enviar mensaje WhatsApp/SMS (si está configurado)")
    print("   ✅ Usar cámara web")
    print("\n" + "=" * 70)
    print("\nPresiona ENTER para iniciar o Ctrl+C para cancelar...")
    print("=" * 70)
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n❌ Cancelado")
        return
    
    try:
        system = PersonDetectionSystem()
        system.run()
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()