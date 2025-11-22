# test_model.py
"""
Script de prueba del modelo YOLOv8
"""

import cv2
import sys
import time
import argparse
from pathlib import Path


def header(text):
    print("\n" + "=" * 70)
    print(f"  {text}".center(70))
    print("=" * 70)


def test_webcam(model_path, confidence=0.5):
    """Prueba con cámara web"""
    header("🧪 TEST - CÁMARA WEB")
    
    # Verificar modelo
    if not model_path.exists():
        print(f"\n❌ Modelo no encontrado: {model_path}")
        print("\n📝 Solución:")
        print("   1. Entrena en Google Colab")
        print("   2. Descarga best.pt")
        print("   3. Colócalo en models/")
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
    print("   Q - Salir")
    print("   S - Screenshot")
    print("   + - Aumentar confianza")
    print("   - - Disminuir confianza")
    print("   P - Pausar")
    print("=" * 70)
    
    paused = False
    fps = 0
    fps_time = time.time()
    fps_count = 0
    screenshot_num = 0
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
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
                
                # Dibujar
                annotated = results[0].plot()
                
                # Contar por clase
                counts = {}
                for box in results[0].boxes:
                    cls = model.names[int(box.cls[0])]
                    counts[cls] = counts.get(cls, 0) + 1
            else:
                annotated = frame.copy()
                inf_ms = 0
            
            # Info panel
            panel_h = 120
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (400, panel_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
            
            info = [
                f"Modelo: {model_path.name}",
                f"Confianza: {confidence:.2f}",
                f"Detecciones: {len(results[0].boxes) if not paused else 0}",
                f"FPS: {fps} | Inf: {inf_ms:.1f}ms",
            ]
            
            if paused:
                info.append("PAUSADO")
            
            y = 25
            for text in info:
                color = (0, 255, 255) if "PAUSADO" not in text else (0, 165, 255)
                cv2.putText(annotated, text, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y += 25
            
            # Mostrar
            cv2.imshow('TEST Detección de Cascos (Q para salir)', annotated)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_num += 1
                cv2.imwrite(f'test_screenshot_{screenshot_num}.jpg', annotated)
                print(f"📸 Screenshot {screenshot_num} guardado")
            elif key in [ord('+'), ord('=')]:
                confidence = min(0.95, confidence + 0.05)
                print(f"🔼 Confianza: {confidence:.2f}")
            elif key in [ord('-'), ord('_')]:
                confidence = max(0.05, confidence - 0.05)
                print(f"🔽 Confianza: {confidence:.2f}")
            elif key == ord('p'):
                paused = not paused
                print(f"{'⏸️ Pausado' if paused else '▶️ Reanudado'}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrumpido")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Prueba completada")


def test_image(model_path, image_path, confidence=0.5):
    """Prueba con imagen"""
    header("🖼️ TEST - IMAGEN")
    
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        return
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Imagen no encontrada: {image_path}")
        return
    
    print(f"✅ Modelo: {model_path}")
    print(f"✅ Imagen: {image_path}")
    
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    
    print("\n📸 Procesando...")
    t0 = time.time()
    results = model(str(image_path), conf=confidence)
    inf_ms = (time.time() - t0) * 1000
    
    boxes = results[0].boxes
    print(f"\n🎯 Resultados:")
    print(f"   Detecciones: {len(boxes)}")
    print(f"   Tiempo: {inf_ms:.1f}ms")
    
    if len(boxes) > 0:
        print("\n📊 Detecciones:")
        for i, box in enumerate(boxes, 1):
            cls = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            print(f"   {i}. {cls}: {conf:.2%}")
    
    # Mostrar
    annotated = results[0].plot()
    cv2.imshow('Resultado - Presiona tecla para cerrar', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Guardar
    output = f"result_{image_path.stem}.jpg"
    cv2.imwrite(output, annotated)
    print(f"\n💾 Guardado: {output}")


def test_video(model_path, video_path, confidence=0.5, save=False):
    """Prueba con video"""
    header("🎬 TEST - VIDEO")
    
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        return
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Video no encontrado: {video_path}")
        return
    
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ Error abriendo video")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {w}x{h} @ {fps}fps, {total} frames")
    
    out = None
    if save:
        output = f"result_{video_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output, fourcc, fps, (w, h))
        print(f"💾 Guardando en: {output}")
    
    print("\n⏳ Procesando (Q para detener)...")
    
    frame_num = 0
    t0 = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            results = model(frame, conf=confidence, verbose=False)
            annotated = results[0].plot()
            
            # Progreso
            pct = (frame_num / total) * 100
            cv2.putText(annotated, f"{frame_num}/{total} ({pct:.1f}%)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Procesando Video (Q para detener)', annotated)
            
            if out:
                out.write(annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if frame_num % 30 == 0:
                print(f"   {frame_num}/{total} ({pct:.1f}%)")
    
    finally:
        elapsed = time.time() - t0
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Completado")
        print(f"   Frames: {frame_num}/{total}")
        print(f"   Tiempo: {elapsed:.1f}s")
        print(f"   FPS: {frame_num/elapsed:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Test del modelo YOLOv8')
    parser.add_argument('input', nargs='?', help='Imagen o video')
    parser.add_argument('--model', default='models/best.pt', help='Ruta al modelo')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confianza')
    parser.add_argument('--video', action='store_true', help='Procesar como video')
    parser.add_argument('--save', action='store_true', help='Guardar resultado')
    
    args = parser.parse_args()
    model_path = Path(args.model)
    
    if not args.input:
        test_webcam(model_path, args.confidence)
    else:
        input_path = Path(args.input)
        ext = input_path.suffix.lower()
        
        if args.video or ext in ['.mp4', '.avi', '.mov', '.mkv']:
            test_video(model_path, input_path, args.confidence, args.save)
        else:
            test_image(model_path, input_path, args.confidence)
    
    print("\n💡 Si funciona aquí, ejecuta: python main.py")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        header("📖 USO")
        print("""
🎥 Cámara web:
   python test_model.py
   python test_model.py --confidence 0.6

🖼️ Imagen:
   python test_model.py imagen.jpg

🎬 Video:
   python test_model.py video.mp4 --video --save

📝 Opciones:
   --model PATH       Ruta al modelo
   --confidence FLOAT Confianza mínima (0-1)
   --video            Procesar como video
   --save             Guardar resultado
""")
    else:
        try:
            main()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()