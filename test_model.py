# test_model.py
"""
Script de prueba del modelo YOLOv8 entrenado
Versión mejorada con más opciones y validaciones
"""

import cv2
from ultralytics import YOLO
import sys
from pathlib import Path
import time
import argparse

def print_header(text):
    """Imprime un encabezado"""
    print("\n" + "="*80)
    print(f"  {text}".center(80))
    print("="*80)

def test_model_webcam(model_path, confidence=0.5):
    """Prueba el modelo con la cámara web"""
    
    print_header("🧪 TEST DEL MODELO - CÁMARA WEB")
    
    # Verificar que existe el modelo
    if not model_path.exists():
        print(f"\n❌ ERROR: No se encontró el modelo")
        print(f"   Buscando en: {model_path.absolute()}")
        print("\n📝 Solución:")
        print("   1. Entrena el modelo en Google Colab")
        print("   2. Descarga best.pt")
        print("   3. Colócalo en la carpeta models/")
        return
    
    print(f"✅ Modelo encontrado: {model_path}")
    size_mb = model_path.stat().st_size / (1024*1024)
    print(f"   Tamaño: {size_mb:.1f} MB")
    
    # Cargar modelo
    print("\n📦 Cargando modelo YOLOv8...")
    try:
        model = YOLO(str(model_path))
        print("✅ Modelo cargado exitosamente")
        
        # Mostrar información del modelo
        print(f"\n📊 Información del modelo:")
        print(f"   Clases: {list(model.names.values())}")
        print(f"   Número de clases: {len(model.names)}")
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    # Abrir cámara
    print("\n🎥 Abriendo cámara web...")
    
    # Probar diferentes índices de cámara
    cap = None
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"✅ Cámara abierta (índice: {camera_index})")
            break
        cap.release()
    
    if not cap or not cap.isOpened():
        print("❌ Error: No se pudo abrir ninguna cámara")
        print("\n📝 Soluciones:")
        print("   1. Verifica que la cámara esté conectada")
        print("   2. Cierra otras aplicaciones que usen la cámara")
        print("   3. Verifica permisos de cámara en tu sistema")
        print("   4. Prueba con una imagen: python test_model.py imagen.jpg")
        return
    
    # Configurar cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Resolución: {actual_width}x{actual_height}")
    
    print_header("🎯 PRUEBA EN VIVO")
    print("📹 Ventana de prueba abierta")
    print("🎮 Controles:")
    print("   • 'q' - Salir")
    print("   • 's' - Capturar screenshot")
    print("   • '+' - Aumentar confianza (+0.05)")
    print("   • '-' - Disminuir confianza (-0.05)")
    print("   • 'i' - Mostrar/ocultar info")
    print("   • 'p' - Pausar/Reanudar")
    print("="*80)
    
    screenshot_count = 0
    show_info = True
    paused = False
    fps = 0
    fps_time = time.time()
    fps_frame_count = 0
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error capturando frame")
                    break
                
                # Calcular FPS
                fps_frame_count += 1
                if time.time() - fps_time >= 1.0:
                    fps = fps_frame_count
                    fps_frame_count = 0
                    fps_time = time.time()
                
                # Realizar detección
                start_time = time.time()
                results = model(frame, conf=confidence, verbose=False)
                inference_time = (time.time() - start_time) * 1000  # en ms
                
                # Dibujar resultados
                annotated_frame = results[0].plot()
                
                # Contar detecciones por clase
                detections = {}
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    detections[class_name] = detections.get(class_name, 0) + 1
            else:
                annotated_frame = frame.copy()
            
            # Panel de información
            if show_info:
                h, w = annotated_frame.shape[:2]
                
                # Fondo del panel
                panel_height = 150 if not paused else 180
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 0), (450, panel_height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                
                # Información
                y_offset = 25
                info_texts = [
                    f"Modelo: {model_path.name}",
                    f"Confianza: {confidence:.2f}",
                    f"Detecciones: {len(results[0].boxes) if not paused else 0}",
                    f"FPS: {fps}",
                    f"Inferencia: {inference_time:.1f}ms" if not paused else "Inferencia: --",
                ]
                
                if paused:
                    info_texts.append("PAUSADO - Presiona 'p'")
                
                for text in info_texts:
                    color = (0, 255, 255) if 'PAUSADO' not in text else (0, 165, 255)
                    cv2.putText(annotated_frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 25
                
                # Detecciones por clase
                if not paused and detections:
                    y_pos = h - 80
                    for class_name, count in detections.items():
                        # Color según clase
                        color = (0, 255, 0) if 'casco' in class_name.lower() and 'sin' not in class_name.lower() else (0, 0, 255)
                        text = f"{class_name}: {count}"
                        cv2.putText(annotated_frame, text, (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        y_pos += 30
            
            # Mostrar frame
            window_name = 'TEST - Detección de Cascos (Presiona Q para salir)'
            cv2.imshow(window_name, annotated_frame)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n✅ Saliendo...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f'screenshot_{screenshot_count}.jpg'
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Screenshot guardado: {filename}")
            elif key == ord('+') or key == ord('='):
                confidence = min(0.95, confidence + 0.05)
                print(f"🔼 Confianza: {confidence:.2f}")
            elif key == ord('-') or key == ord('_'):
                confidence = max(0.05, confidence - 0.05)
                print(f"🔽 Confianza: {confidence:.2f}")
            elif key == ord('i'):
                show_info = not show_info
                print(f"ℹ️  Info: {'ON' if show_info else 'OFF'}")
            elif key == ord('p'):
                paused = not paused
                print(f"{'⏸️  Pausado' if paused else '▶️  Reanudado'}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido por el usuario")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Prueba completada")
        print("="*80)

def test_model_on_image(model_path, image_path, confidence=0.5, save_result=True):
    """Prueba el modelo en una imagen específica"""
    
    print_header("🖼️  TEST DEL MODELO - IMAGEN ESTÁTICA")
    
    # Verificar modelo
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        return
    
    # Verificar imagen
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Imagen no encontrada: {image_path}")
        return
    
    print(f"✅ Modelo: {model_path}")
    print(f"✅ Imagen: {image_path}")
    
    # Cargar modelo
    print("\n📦 Cargando modelo...")
    try:
        model = YOLO(str(model_path))
        print("✅ Modelo cargado")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Realizar predicción
    print(f"\n📸 Procesando imagen...")
    start_time = time.time()
    results = model(str(image_path), conf=confidence)
    inference_time = (time.time() - start_time) * 1000
    
    # Obtener imagen anotada
    annotated = results[0].plot()
    
    # Información
    boxes = results[0].boxes
    print(f"\n🎯 Resultados:")
    print(f"   Detecciones: {len(boxes)}")
    print(f"   Tiempo de inferencia: {inference_time:.1f}ms")
    
    if len(boxes) > 0:
        print("\n📊 Detecciones encontradas:")
        print("─"*80)
        
        for i, box in enumerate(boxes, 1):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            
            print(f"   {i}. {class_name}")
            print(f"      Confianza: {conf:.2%}")
            print(f"      Ubicación: ({x1},{y1}) - ({x2},{y2})")
            print(f"      Área: {area:,}px²")
    else:
        print("\n⚠️  No se detectaron objetos")
        print(f"   (Umbral de confianza: {confidence})")
    
    # Mostrar imagen
    print("\n🖼️  Mostrando resultado...")
    cv2.imshow('Resultado - Presiona cualquier tecla para cerrar', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Guardar resultado
    if save_result:
        output_name = f"result_{image_path.stem}.jpg"
        cv2.imwrite(output_name, annotated)
        print(f"\n💾 Resultado guardado: {output_name}")
    
    print("="*80)

def test_model_on_video(model_path, video_path, confidence=0.5, save_result=False):
    """Prueba el modelo en un video"""
    
    print_header("🎬 TEST DEL MODELO - VIDEO")
    
    # Verificar archivos
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        return
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Video no encontrado: {video_path}")
        return
    
    print(f"✅ Modelo: {model_path}")
    print(f"✅ Video: {video_path}")
    
    # Cargar modelo
    model = YOLO(str(model_path))
    
    # Abrir video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print("❌ Error abriendo video")
        return
    
    # Propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Propiedades del video:")
    print(f"   Resolución: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Frames totales: {total_frames}")
    print(f"   Duración: {total_frames/fps:.1f}s")
    
    # Preparar salida si se solicita
    out = None
    if save_result:
        output_name = f"result_{video_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
        print(f"\n💾 Guardando resultado en: {output_name}")
    
    print("\n⏳ Procesando video...")
    print("   Presiona 'q' para detener")
    
    frame_count = 0
    total_detections = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Procesar cada N frames para ser más rápido
            if frame_count % 2 == 0:  # Procesar cada 2 frames
                results = model(frame, conf=confidence, verbose=False)
                annotated = results[0].plot()
                total_detections += len(results[0].boxes)
            else:
                annotated = frame
            
            # Agregar información
            progress = (frame_count / total_frames) * 100
            text = f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)"
            cv2.putText(annotated, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar
            cv2.imshow('Procesando Video - Presiona Q para detener', annotated)
            
            # Guardar
            if out:
                out.write(annotated)
            
            # Controles
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠️  Detenido por el usuario")
                break
            
            # Progreso
            if frame_count % 30 == 0:
                print(f"   Procesado: {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido")
    
    finally:
        elapsed = time.time() - start_time
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Procesamiento completado")
        print(f"   Frames procesados: {frame_count}/{total_frames}")
        print(f"   Tiempo: {elapsed:.1f}s")
        print(f"   FPS promedio: {frame_count/elapsed:.1f}")
        print(f"   Detecciones totales: {total_detections}")
        
        if save_result:
            print(f"   Video guardado: {output_name}")
        
        print("="*80)

def print_usage():
    """Muestra instrucciones de uso"""
    print_header("📖 GUÍA DE USO - TEST_MODEL.PY")
    
    print("""
🎥 Probar con cámara web:
   python test_model.py
   python test_model.py --confidence 0.6

🖼️  Probar con imagen:
   python test_model.py imagen.jpg
   python test_model.py foto.jpg --confidence 0.7

🎬 Probar con video:
   python test_model.py video.mp4 --video
   python test_model.py video.mp4 --video --save

📝 Opciones:
   --model PATH       Ruta al modelo (default: models/best.pt)
   --confidence FLOAT Confianza mínima (default: 0.5)
   --video            Procesar como video
   --save             Guardar resultado
   --help             Mostrar esta ayuda

💡 Ejemplos:
   python test_model.py
   python test_model.py imagen.jpg
   python test_model.py video.mp4 --video --save
   python test_model.py --model models/yolov8n.pt --confidence 0.7
""")
    
    print("="*80)

def main():
    """Función principal"""
    
    # Argumentos
    parser = argparse.ArgumentParser(description='Test del modelo YOLOv8')
    parser.add_argument('input', nargs='?', help='Imagen o video de entrada')
    parser.add_argument('--model', default='models/best.pt', help='Ruta al modelo')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confianza mínima')
    parser.add_argument('--video', action='store_true', help='Procesar como video')
    parser.add_argument('--save', action='store_true', help='Guardar resultado')
    
    args = parser.parse_args()
    
    # Ruta al modelo
    model_path = Path(args.model)
    
    # Sin input = cámara web
    if not args.input:
        test_model_webcam(model_path, args.confidence)
    
    # Con input
    else:
        input_path = Path(args.input)
        
        # Video
        if args.video or input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            test_model_on_video(model_path, input_path, args.confidence, args.save)
        
        # Imagen
        else:
            test_model_on_image(model_path, input_path, args.confidence, args.save)
    
    print("\n💡 TIP: Si todo funciona bien aquí, el sistema principal también funcionará")
    print("   Ejecuta: python main.py")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print_usage()
    else:
        try:
            main()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)