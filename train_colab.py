# ============================================================
# ENTRENAMIENTO YOLOV8 - DETECCIÓN DE CASCOS
# Google Colab - Versión Completa y Optimizada
# ============================================================
# INSTRUCCIONES:
# 1. Sube este archivo a Google Colab
# 2. Activa GPU: Runtime → Change runtime type → GPU
# 3. Ejecuta cada celda en orden (Shift+Enter)
# ============================================================

# %% [markdown]
# # 🛡️ Entrenamiento YOLOv8 - Detección de Cascos
# ## SENATI - Ingeniería Civil

# %% [markdown]
# ## CELDA 1: Verificar GPU y Entorno

# %%
import torch
import sys
import os

print("=" * 80)
print("🔍 VERIFICACIÓN DEL ENTORNO".center(80))
print("=" * 80)

# Versión de Python
py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
print(f"🐍 Python: {py_ver}")
print(f"✅ PyTorch: {torch.__version__}")

# CUDA
cuda_ok = torch.cuda.is_available()
print(f"{'✅' if cuda_ok else '❌'} CUDA disponible: {cuda_ok}")

if cuda_ok:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ Memoria GPU: {gpu_mem:.2f} GB")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print("\n🎉 ¡GPU lista para entrenar!")
else:
    print("\n" + "!" * 80)
    print("⚠️  ¡ATENCIÓN! GPU NO DETECTADA".center(80))
    print("!" * 80)
    print("""
🔧 CÓMO ACTIVAR GPU EN COLAB:
   1. Click en 'Runtime' (Entorno de ejecución)
   2. Click en 'Change runtime type'
   3. En 'Hardware accelerator' selecciona 'GPU'
   4. Click 'Save'
   5. El notebook se reiniciará
   6. Re-ejecuta esta celda

💡 Sin GPU: ~10 horas | Con GPU: ~1-2 horas
""")

print("=" * 80)

# %% [markdown]
# ## CELDA 2: Instalar Dependencias

# %%
print("\n📦 Instalando dependencias...")
print("=" * 80)

# Instalar paquetes necesarios
!pip install -q ultralytics roboflow

# Verificar instalación
try:
    from ultralytics import YOLO
    print("✅ Ultralytics (YOLOv8) instalado")
except ImportError as e:
    print(f"❌ Error: {e}")

try:
    from roboflow import Roboflow
    print("✅ Roboflow instalado")
except ImportError as e:
    print(f"❌ Error: {e}")

import ultralytics
print(f"   Versión Ultralytics: {ultralytics.__version__}")
print("=" * 80)

# %% [markdown]
# ## CELDA 3: Montar Google Drive

# %%
print("\n☁️  Montando Google Drive...")
print("=" * 80)

from google.colab import drive
from pathlib import Path

try:
    drive.mount('/content/drive', force_remount=False)
    print("✅ Google Drive montado")
    
    # Crear carpeta del proyecto
    project_dir = Path('/content/drive/MyDrive/helmet_detection')
    project_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Carpeta del proyecto: {project_dir}")
    
    # Verificar espacio
    stat = os.statvfs('/content/drive/MyDrive')
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    print(f"💾 Espacio disponible: {free_gb:.1f} GB")
    
    if free_gb < 1:
        print("⚠️  Poco espacio en Drive. Libera espacio antes de continuar.")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("Autoriza el acceso cuando se solicite")

print("=" * 80)

# %% [markdown]
# ## CELDA 4: Descargar Dataset de Roboflow
# 
# **IMPORTANTE:** Si tienes tu propio dataset, modifica las credenciales abajo.

# %%
print("\n📥 DESCARGANDO DATASET DE ROBOFLOW")
print("=" * 80)

from roboflow import Roboflow

# ============================================================
# 🔧 CONFIGURACIÓN DEL DATASET - MODIFICA AQUÍ SI ES NECESARIO
# ============================================================
ROBOFLOW_API_KEY = "oE6oO2EQShRDIYIeUfkS"  # Tu API key
WORKSPACE_NAME = "cosmiko"                   # Tu workspace
PROJECT_NAME = "deteccion-de-cascos-de-seguridad-028dh"  # Tu proyecto
VERSION_NUMBER = 1                           # Versión del dataset
# ============================================================

try:
    # Conectar a Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    print(f"✅ Conectado a Roboflow")
    
    # Obtener proyecto y versión
    project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
    print(f"✅ Proyecto: {PROJECT_NAME}")
    
    version = project.version(VERSION_NUMBER)
    print(f"✅ Versión: {VERSION_NUMBER}")
    
    # Descargar en formato YOLOv8
    print("\n⏳ Descargando dataset (1-3 minutos)...")
    dataset = version.download("yolov8")
    
    print(f"\n✅ Dataset descargado exitosamente")
    print(f"📁 Ubicación: {dataset.location}")
    
except Exception as e:
    print(f"\n❌ Error descargando dataset: {e}")
    print("""
📝 Soluciones:
   • Verifica tu API key de Roboflow
   • Comprueba el nombre del workspace y proyecto
   • Asegúrate de tener conexión a internet
   
💡 Para obtener tu API key:
   1. Ve a roboflow.com y crea cuenta
   2. Settings → API Key → Copiar
   3. Reemplaza ROBOFLOW_API_KEY arriba
""")
    raise

print("=" * 80)

# %% [markdown]
# ## CELDA 5: Verificar Estructura del Dataset

# %%
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import random

print("\n📊 ESTRUCTURA DEL DATASET")
print("=" * 80)

dataset_path = Path(dataset.location)

# Contar imágenes
def count_images(folder):
    path = dataset_path / folder / 'images'
    if path.exists():
        return len(list(path.glob('*')))
    return 0

train_count = count_images('train')
valid_count = count_images('valid')
test_count = count_images('test')

print(f"📁 Train:  {train_count:>4} imágenes")
print(f"📁 Valid:  {valid_count:>4} imágenes")
print(f"📁 Test:   {test_count:>4} imágenes")
print(f"{'─' * 40}")
print(f"📄 Total:  {train_count + valid_count + test_count:>4} imágenes")

# Advertencia si hay pocas imágenes
if train_count < 100:
    print(f"\n⚠️  Solo {train_count} imágenes de entrenamiento")
    print("   Recomendado: 200+ para mejores resultados")

# Verificar data.yaml
data_yaml_path = dataset_path / 'data.yaml'

if data_yaml_path.exists():
    print(f"\n✅ Archivo data.yaml encontrado")
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("\n📄 Configuración del dataset:")
    print("─" * 80)
    
    # Mostrar info relevante
    for key in ['train', 'val', 'test', 'nc', 'names']:
        if key in data_config:
            print(f"   {key}: {data_config[key]}")
    
    print("─" * 80)
    
    class_names = data_config.get('names', [])
    num_classes = data_config.get('nc', len(class_names))
    
    print(f"\n📊 Clases del modelo:")
    print(f"   Cantidad: {num_classes}")
    for i, name in enumerate(class_names):
        print(f"   {i}: {name}")
else:
    print("\n❌ ERROR: data.yaml no encontrado")
    raise FileNotFoundError("data.yaml no encontrado en el dataset")

print("=" * 80)

# %% [markdown]
# ## CELDA 6: Visualizar Muestras del Dataset

# %%
print("\n🖼️  VISUALIZANDO MUESTRAS DEL DATASET")
print("=" * 80)

def draw_boxes(img_path, label_path, class_names):
    """Dibuja bounding boxes en la imagen"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_c, y_c, bw, bh = map(float, parts[1:5])
                    
                    # Convertir a píxeles
                    x1 = int((x_c - bw/2) * w)
                    y1 = int((y_c - bh/2) * h)
                    x2 = int((x_c + bw/2) * w)
                    y2 = int((y_c + bh/2) * h)
                    
                    # Color por clase
                    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
                    color = colors[cls % len(colors)]
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Etiqueta
                    label = class_names[cls] if cls < len(class_names) else f"C{cls}"
                    cv2.putText(img, label, (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

# Obtener imágenes de entrenamiento
train_imgs_path = dataset_path / 'train' / 'images'
train_images = list(train_imgs_path.glob('*'))

if len(train_images) >= 6:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    samples = random.sample(train_images, 6)
    
    for i, img_path in enumerate(samples):
        label_path = dataset_path / 'train' / 'labels' / (img_path.stem + '.txt')
        img = draw_boxes(img_path, label_path, class_names)
        
        if img is not None:
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f'Muestra {i+1}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/content/dataset_samples.png', dpi=150)
    plt.show()
    print("✅ Muestras visualizadas")
else:
    print("⚠️  Muy pocas imágenes para visualizar")

print("=" * 80)

# %% [markdown]
# ## CELDA 7: Checklist Pre-Entrenamiento

# %%
print("\n✅ CHECKLIST PRE-ENTRENAMIENTO")
print("=" * 80)

checks = {
    "GPU disponible": torch.cuda.is_available(),
    "Dataset descargado": dataset_path.exists(),
    "data.yaml existe": data_yaml_path.exists(),
    "Imágenes train > 50": train_count > 50,
    "Imágenes valid > 10": valid_count > 10,
    "Google Drive montado": Path('/content/drive/MyDrive').exists(),
}

all_ok = True
for check, status in checks.items():
    icon = "✅" if status else "❌"
    print(f"{icon} {check}")
    if not status:
        all_ok = False

print("=" * 80)

if all_ok:
    print("\n🎉 ¡Todo listo para entrenar!")
    print("Ejecuta la siguiente celda para iniciar")
else:
    print("\n⚠️  Resuelve los problemas marcados con ❌")
    print("No continúes hasta que todo esté ✅")

# %% [markdown]
# ## CELDA 8: 🚀 ENTRENAR EL MODELO
# 
# **Esta celda tomará 1-2 horas con GPU**

# %%
print("\n🚀 INICIANDO ENTRENAMIENTO")
print("=" * 80)

from ultralytics import YOLO
import time

# Verificar dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Dispositivo: {device.upper()}")

if device == 'cpu':
    print("\n⚠️  ADVERTENCIA: Entrenando en CPU (muy lento)")
    confirm = input("¿Continuar? (escribe 'si'): ").strip().lower()
    if confirm != 'si':
        raise SystemExit("Activa GPU y reinicia")
else:
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   Memoria: {mem_gb:.1f} GB")

# Cargar modelo base
print("\n📦 Cargando modelo base YOLOv8n...")
model = YOLO('yolov8n.pt')
print("✅ Modelo base cargado")

# Configuración del entrenamiento
print("\n⚙️  Configuración:")
print("─" * 80)

# Ajustar batch size según GPU
if device == 'cuda':
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    batch_size = 16 if mem_gb >= 10 else 8
else:
    batch_size = 4

config = {
    'data': str(data_yaml_path),
    'epochs': 100,
    'imgsz': 640,
    'batch': batch_size,
    'name': 'helmet_detection',
    'patience': 50,
    'save': True,
    'device': device,
    'workers': 4,
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'plots': True,
    'save_period': 10,
    'verbose': True,
    'project': '/content/runs/detect',
}

for k, v in config.items():
    print(f"   {k:<18}: {v}")
print("─" * 80)

# Estimación de tiempo
est_time = (1.5 * config['epochs']) / 100 if device == 'cuda' else 10
print(f"\n⏱️  Tiempo estimado: ~{est_time:.1f} horas")
print("\n🎯 ENTRENAMIENTO INICIADO...")
print("   No cierres esta ventana\n")
print("=" * 80 + "\n")

start_time = time.time()

try:
    results = model.train(**config)
    
    duration_min = (time.time() - start_time) / 60
    duration_hr = duration_min / 60
    
    print("\n" + "=" * 80)
    print("✅ ENTRENAMIENTO COMPLETADO".center(80))
    print("=" * 80)
    print(f"⏱️  Tiempo total: {duration_min:.1f} min ({duration_hr:.2f} hrs)")
    print("=" * 80)
    
except KeyboardInterrupt:
    print("\n⚠️  Entrenamiento interrumpido")
    print(f"   Pesos guardados en: {config['project']}/{config['name']}/weights/")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("""
📝 Soluciones comunes:
   • Reduce batch_size a 4 o 8
   • Verifica que data.yaml tenga rutas correctas
   • Reinicia el runtime y vuelve a ejecutar
""")
    raise

# %% [markdown]
# ## CELDA 9: Evaluar el Modelo

# %%
print("\n📊 EVALUANDO MODELO")
print("=" * 80)

# Ruta al mejor modelo
best_model_path = f'/content/runs/detect/{config["name"]}/weights/best.pt'

if not Path(best_model_path).exists():
    # Buscar en carpetas alternativas
    possible_paths = list(Path('/content/runs/detect').glob('*/weights/best.pt'))
    if possible_paths:
        best_model_path = str(possible_paths[-1])
        print(f"📍 Modelo encontrado en: {best_model_path}")
    else:
        print("❌ No se encontró el modelo entrenado")
        raise FileNotFoundError("best.pt no encontrado")

print(f"📦 Cargando: {best_model_path}")
best_model = YOLO(best_model_path)
print("✅ Modelo cargado")

# Evaluar
print("\n⏳ Evaluando en conjunto de validación...")
metrics = best_model.val()

# Métricas
print("\n📈 MÉTRICAS DEL MODELO:")
print("=" * 80)
print(f"{'Métrica':<20} {'Valor':<12} {'Descripción'}")
print("─" * 80)
print(f"{'mAP50':<20} {metrics.box.map50:.4f}       Precisión al 50% IoU")
print(f"{'mAP50-95':<20} {metrics.box.map:.4f}       Precisión promedio")
print(f"{'Precision':<20} {metrics.box.mp:.4f}       TP / (TP+FP)")
print(f"{'Recall':<20} {metrics.box.mr:.4f}       TP / (TP+FN)")
print("=" * 80)

# Interpretación
map50 = metrics.box.map50
print("\n💡 INTERPRETACIÓN:")
if map50 > 0.9:
    print("   🌟 EXCELENTE (>90%) - Modelo muy preciso")
elif map50 > 0.8:
    print("   ✅ MUY BUENO (80-90%) - Listo para producción")
elif map50 > 0.7:
    print("   👍 BUENO (70-80%) - Funcional, puede mejorar")
elif map50 > 0.5:
    print("   ⚠️  ACEPTABLE (50-70%) - Necesita más datos")
else:
    print("   ❌ INSUFICIENTE (<50%) - Re-entrenar con más datos")

print("=" * 80)

# %% [markdown]
# ## CELDA 10: Visualizar Resultados

# %%
from IPython.display import Image, display

print("\n📊 VISUALIZACIÓN DE RESULTADOS")
print("=" * 80)

results_dir = Path(best_model_path).parent.parent

# Lista de gráficos a mostrar
plots = [
    ('results.png', 'Curvas de Entrenamiento'),
    ('confusion_matrix.png', 'Matriz de Confusión'),
    ('PR_curve.png', 'Curva Precision-Recall'),
    ('val_batch0_pred.jpg', 'Predicciones en Validación'),
]

for filename, title in plots:
    filepath = results_dir / filename
    print(f"\n📈 {title}:")
    print("─" * 80)
    
    if filepath.exists():
        display(Image(str(filepath), width=800))
        print("✅ Mostrado")
    else:
        print(f"⚠️  No encontrado: {filename}")

print("=" * 80)

# %% [markdown]
# ## CELDA 11: Probar con Imágenes

# %%
print("\n🧪 PROBANDO MODELO")
print("=" * 80)

# Imágenes de validación
valid_imgs = list((dataset_path / 'valid' / 'images').glob('*'))

if len(valid_imgs) >= 6:
    samples = random.sample(valid_imgs, 6)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, img_path in enumerate(samples):
        results = best_model(str(img_path), conf=0.5, verbose=False)
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(annotated)
        axes[i].axis('off')
        n_det = len(results[0].boxes)
        axes[i].set_title(f'{n_det} detecciones', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('/content/model_predictions.png', dpi=150)
    plt.show()
    print("✅ Predicciones mostradas")
else:
    print("⚠️  Pocas imágenes de validación")

print("=" * 80)

# %% [markdown]
# ## CELDA 12: 💾 GUARDAR EN GOOGLE DRIVE

# %%
print("\n💾 GUARDANDO EN GOOGLE DRIVE")
print("=" * 80)

import shutil
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Rutas
src_model = best_model_path
drive_dir = Path('/content/drive/MyDrive/helmet_detection')

# Crear carpetas
(drive_dir / 'models').mkdir(parents=True, exist_ok=True)
(drive_dir / 'results').mkdir(parents=True, exist_ok=True)

try:
    # 1. Modelo principal
    dst_best = drive_dir / 'models' / 'best.pt'
    shutil.copy(src_model, dst_best)
    size_mb = dst_best.stat().st_size / (1024*1024)
    print(f"✅ Modelo guardado: best.pt ({size_mb:.1f} MB)")
    
    # 2. Backup con timestamp
    dst_backup = drive_dir / 'models' / f'best_{timestamp}.pt'
    shutil.copy(src_model, dst_backup)
    print(f"✅ Backup: best_{timestamp}.pt")
    
    # 3. Último checkpoint
    last_pt = Path(src_model).parent / 'last.pt'
    if last_pt.exists():
        shutil.copy(last_pt, drive_dir / 'models' / 'last.pt')
        print(f"✅ Último checkpoint: last.pt")
    
    # 4. Resultados del entrenamiento
    src_results = Path(src_model).parent.parent
    dst_results = drive_dir / 'results' / f'training_{timestamp}'
    if src_results.exists():
        shutil.copytree(src_results, dst_results, dirs_exist_ok=True)
        print(f"✅ Resultados: training_{timestamp}/")
    
    # 5. Info del entrenamiento
    info_file = drive_dir / 'models' / f'info_{timestamp}.txt'
    with open(info_file, 'w') as f:
        f.write(f"Entrenamiento de Detección de Cascos\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duración: {duration_min:.1f} minutos\n")
        f.write(f"Épocas: {config['epochs']}\n")
        f.write(f"Batch: {config['batch']}\n")
        f.write(f"Imágenes train: {train_count}\n")
        f.write(f"Imágenes valid: {valid_count}\n\n")
        f.write(f"Métricas:\n")
        f.write(f"  mAP50: {metrics.box.map50:.4f}\n")
        f.write(f"  mAP50-95: {metrics.box.map:.4f}\n")
        f.write(f"  Precision: {metrics.box.mp:.4f}\n")
        f.write(f"  Recall: {metrics.box.mr:.4f}\n")
    print(f"✅ Info: info_{timestamp}.txt")
    
    print(f"""
📁 Estructura en Google Drive:
   helmet_detection/
   ├── models/
   │   ├── best.pt ← DESCARGA ESTE
   │   ├── best_{timestamp}.pt
   │   ├── last.pt
   │   └── info_{timestamp}.txt
   └── results/
       └── training_{timestamp}/

📥 SIGUIENTE PASO:
   1. Ve a Google Drive
   2. Abre: helmet_detection/models/
   3. Descarga: best.pt
   4. Colócalo en: tu_proyecto/models/best.pt
""")
    
except Exception as e:
    print(f"❌ Error guardando: {e}")
    print(f"\n📍 Copia manualmente desde:")
    print(f"   {src_model}")

print("=" * 80)

# %% [markdown]
# ## CELDA 13: Descargar Modelo Directamente

# %%
from google.colab import files

print("\n📥 DESCARGAR MODELO DIRECTAMENTE")
print("=" * 80)

# Descargar best.pt
if Path(best_model_path).exists():
    print("📦 Preparando descarga de best.pt...")
    try:
        files.download(best_model_path)
        print("✅ Descarga iniciada")
        print("\n💡 Si no inicia la descarga automáticamente:")
        print(f"   1. Ve a: /content/runs/detect/{config['name']}/weights/")
        print("   2. Click derecho en best.pt")
        print("   3. Descargar")
    except Exception as e:
        print(f"⚠️  Descarga manual: {e}")
        print(f"   Archivo en: {best_model_path}")
else:
    print("❌ Modelo no encontrado")

print("\n" + "=" * 80)
print("🎉 ¡ENTRENAMIENTO COMPLETADO!".center(80))
print("=" * 80)
print("""
📋 RESUMEN FINAL:

✅ Modelo entrenado exitosamente
✅ Guardado en Google Drive
✅ Listo para descargar

📥 PRÓXIMOS PASOS:
   1. Descarga best.pt
   2. Colócalo en tu_proyecto/models/
   3. Ejecuta: python main.py

¡Gracias por usar este sistema!
SENATI - Ingeniería Civil
""")
print("=" * 80)