# test_model.py
"""
Script de prueba del modelo YOLOv8 entrenado
Ejecuta esto para verificar que el modelo funciona antes de usar el sistema completo
"""

import cv2
from ultralytics import YOLO
import sys
from pathlib import Path

def test_model():
    """Prueba el modelo con la cámara web"""
    
    print("="*60)
    print("🧪 TEST DEL MODELO YOLOV8 - DETECCIÓN DE CASCOS")
    print("="*60)
    
    # Verificar que existe el modelo
    model_path = Path('models/best.pt')
    
    if not model_path.exists():
        print("\n❌ ERROR: No se encontró el modelo")
        print(f"   Buscando en: {model_path.absolute()}")
        print("\n📝 Solución:")
        print("   1. Entrena el modelo en Google Colab")
        print("   2. Descarga best.pt")
        print("   3. Colócalo en la carpeta models/")
        print("="*60)
        return
    
    print(f"✅ Modelo encontrado: {model_path}")
    
    # Cargar modelo
    print("\n📦 Cargando modelo YOLOv8...")
    try:
        model = YOLO(str(model_path))
        print("✅ Modelo cargado exitosamente")
        
        # Mostrar información del modelo
        print(f"\n📊 Información del modelo:")
        print(f"   Clases: {model.names}")
        print(f"   Número de clases: {len(model.names)}")
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    # Abrir cámara
    print("\n🎥 Abriendo cámara web...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara")
        print("\n📝 Soluciones:")
        print("   1. Verifica que la cámara esté conectada")
        print("   2. Cierra otras aplicaciones que usen la cámara")
        print("   3. Prueba cambiar el índice de cámara:")
        print("      cap = cv2.VideoCapture(1)  # o 2, 3, etc.")
        return
    
    print("✅ Cámara abierta correctamente")
    
    # Configurar cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n" + "="*60)
    print("🎯 PRUEBA EN VIVO")
    print("="*60)
    print("📹 Ventana de prueba abierta")
    print("🎮 Controles:")
    print("   • Presiona 'q' para salir")
    print("   • Presiona 's' para capturar screenshot")
    print("   • Presiona '+' para aumentar confianza")
    print("   • Presiona '-' para disminuir confianza")
    print("="*60)
    
    confidence = 0.5
    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error capturando frame")
                break
            
            # Realizar detección
            results = model(frame, conf=confidence, verbose=False)
            
            # Dibujar resultados
            annotated_frame = results[0].plot()
            
            # Información en pantalla
            h, w = annotated_frame.shape[:2]
            
            # Panel de información
            cv2.rectangle(annotated_frame, (0, 0), (400, 120), (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, (0, 0), (400, 120), (255, 255, 255), 2)
            
            # Texto
            info_text = [
                f"Modelo: YOLOv8",
                f"Confianza: {confidence:.2f}",
                f"Detecciones: {len(results[0].boxes)}",
                f"FPS: {1000/results[0].speed['inference']:.1f}"
            ]
            
            y_offset = 25
            for text in info_text:
                cv2.putText(annotated_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            
            # Contador de detecciones por clase
            detections = {}
            for box in results[0].boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                detections[class_name] = detections.get(class_name, 0) + 1
            
            # Mostrar detecciones por clase
            y_pos = h - 80
            for class_name, count in detections.items():
                color = (0, 255, 0) if 'con' in class_name.lower() else (0, 0, 255)
                text = f"{class_name}: {count}"
                cv2.putText(annotated_frame, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_pos += 30
            
            # Mostrar frame
            cv2.imshow('TEST - Detección de Cascos (Presiona Q para salir)', annotated_frame)
            
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
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido por el usuario")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Prueba completada")
        print("="*60)

def test_model_on_image(image_path):
    """Prueba el modelo en una imagen específica"""
    
    print("\n🖼️  PRUEBA CON IMAGEN ESTÁTICA")
    print("="*60)
    
    model_path = Path('models/best.pt')
    
    if not model_path.exists():
        print("❌ Modelo no encontrado")
        return
    
    if not Path(image_path).exists():
        print(f"❌ Imagen no encontrada: {image_path}")
        return
    
    # Cargar modelo
    model = YOLO(str(model_path))
    
    # Realizar predicción
    print(f"📸 Procesando: {image_path}")
    results = model(image_path, conf=0.5)
    
    # Mostrar resultados
    annotated = results[0].plot()
    
    # Información
    boxes = results[0].boxes
    print(f"\n🎯 Detecciones encontradas: {len(boxes)}")
    
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls]
        print(f"   {i+1}. {class_name}: {conf:.2%}")
    
    # Mostrar imagen
    cv2.imshow('Resultado - Presiona cualquier tecla para cerrar', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Guardar resultado
    output_path = f"result_{Path(image_path).name}"
    cv2.imwrite(output_path, annotated)
    print(f"\n💾 Resultado guardado: {output_path}")
    print("="*60)

def print_usage():
    """Muestra instrucciones de uso"""
    print("\n📖 USO DEL SCRIPT DE PRUEBA")
    print("="*60)
    print("\n🎥 Probar con cámara web:")
    print("   python test_model.py")
    print("\n🖼️  Probar con imagen:")
    print("   python test_model.py imagen.jpg")
    print("\n📝 Ejemplos:")
    print("   python test_model.py")
    print("   python test_model.py foto_casco.jpg")
    print("   python test_model.py C:/imagenes/test.png")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Modo imagen
        if sys.argv[1] in ['--help', '-h', 'help']:
            print_usage()
        else:
            test_model_on_image(sys.argv[1])
    else:
        # Modo cámara web
        test_model()
    
    print("\n💡 TIP: Si todo funciona bien aquí, el sistema principal también funcionará")
    print("   Ejecuta: python main.py")
    print()