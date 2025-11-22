# test_model.py
"""
Script de prueba - VERSIÓN SIMPLIFICADA
Solo detecta CASCOS - Si no hay casco visible = puede haber peligro
"""

import cv2
import sys
import time
from pathlib import Path
import numpy as np


def header(text):
    print("\n" + "=" * 70)
    print(f"  {text}".center(70))
    print("=" * 70)


def test_webcam(model_path, confidence=0.5):
    """Prueba con cámara - SOLO detecta cascos"""
    header("🧪 TEST - DETECCIÓN DE CASCOS")
    
    if not model_path.exists():
        print(f"\n❌ Modelo no encontrado: {model_path}")
        return
    
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✅ Modelo: {model_path.name} ({size_mb:.1f} MB)")
    
    # Cargar modelo
    print("\n📦 Cargando modelo...")
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        print(f"✅ Cargado - Clases: {list(model.names.values())}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Abrir cámara
    print("\n🎥 Abriendo cámara...")
    cap = None
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"✅ Cámara {idx} abierta")
            break
        cap.release()
    
    if not cap or not cap.isOpened():
        print("❌ No se encontró cámara")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   Resolución: {w}x{h}")
    
    header("🎯 PRUEBA EN VIVO")
    print("🎮 Controles:")
    print("   Q/ESC  - Salir")
    print("   S      - Screenshot")
    print("   +/-    - Ajustar confianza")
    print("")
    print("📊 LÓGICA:")
    print("   🟢 CASCO DETECTADO = Seguro")
    print("   🟡 SIN DETECCIÓN   = Verificar área")
    print("=" * 70)
    
    fps = 0
    fps_time = time.time()
    fps_count = 0
    screenshot_num = 0
    
    window_name = 'Deteccion de Cascos - Q/ESC para salir'
    
    # Colores
    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            h_frame, w_frame = frame.shape[:2]
            
            # FPS
            fps_count += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_count
                fps_count = 0
                fps_time = time.time()
            
            # Detección
            t0 = time.time()
            results = model(frame, conf=confidence, verbose=False)
            inf_ms = (time.time() - t0) * 1000
            
            # Procesar detecciones
            cascos_detectados = 0
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Filtrar detecciones muy pequeñas (probablemente falsos positivos)
                box_w = x2 - x1
                box_h = y2 - y1
                area = box_w * box_h
                
                # Ignorar si es muy pequeño o tiene proporciones raras
                if area < 3000:  # Muy pequeño
                    continue
                if box_w / max(box_h, 1) > 3 or box_h / max(box_w, 1) > 3:  # Proporción rara
                    continue
                
                cascos_detectados += 1
                
                # Dibujar casco (VERDE)
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 3)
                label = f"CASCO {conf:.0%}"
                
                # Fondo del label
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1-30), (x1+tw+10, y1), GREEN, -1)
                cv2.putText(frame, label, (x1+5, y1-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
            
            # Panel de información superior
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (320, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Info
            cv2.putText(frame, f"Modelo: {model_path.name}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2)
            cv2.putText(frame, f"Confianza: {confidence:.0%}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2)
            cv2.putText(frame, f"Cascos detectados: {cascos_detectados}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN if cascos_detectados > 0 else YELLOW, 2)
            cv2.putText(frame, f"FPS: {fps} | Inf: {inf_ms:.0f}ms", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)
            
            # Estado en la parte inferior
            if cascos_detectados > 0:
                # Barra verde - SEGURO
                cv2.rectangle(frame, (0, h_frame-50), (w_frame, h_frame), GREEN, -1)
                cv2.putText(frame, f"SEGURO - {cascos_detectados} CASCO(S) DETECTADO(S)",
                           (w_frame//2-180, h_frame-18), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, WHITE, 2)
            else:
                # Barra amarilla - SIN DETECCIÓN
                cv2.rectangle(frame, (0, h_frame-50), (w_frame, h_frame), YELLOW, -1)
                cv2.putText(frame, "VERIFICAR - NO SE DETECTAN CASCOS EN EL AREA",
                           (w_frame//2-220, h_frame-18), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 0, 0), 2)
            
            # Mostrar
            cv2.imshow(window_name, frame)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q') or key == 27:
                print("\n👋 Saliendo...")
                break
            elif key == ord('s') or key == ord('S'):
                screenshot_num += 1
                filename = f'screenshot_{screenshot_num}.jpg'
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot: {filename}")
            elif key in [ord('+'), ord('=')]:
                confidence = min(0.95, confidence + 0.05)
                print(f"🔼 Confianza: {confidence:.0%}")
            elif key in [ord('-'), ord('_')]:
                confidence = max(0.1, confidence - 0.05)
                print(f"🔽 Confianza: {confidence:.0%}")
            
            # Verificar si se cerró la ventana
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except:
                break
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrumpido")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
        print("\n✅ Prueba completada")
        print("\n💡 Este modelo SOLO detecta cascos.")
        print("   Para detectar 'personas sin casco' necesitas un modelo")
        print("   entrenado con DOS clases: 'con_casco' y 'sin_casco'")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/best.pt')
    parser.add_argument('--confidence', type=float, default=0.5)
    args = parser.parse_args()
    
    test_webcam(Path(args.model), args.confidence)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()