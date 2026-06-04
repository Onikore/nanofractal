# nanofractal — высокопроизводительная Python-библиотека детекции маркеров

**Дата:** 2026-06-04
**Статус:** утверждён дизайн, готов к планированию реализации

## 1. Цель

Создать Python-библиотеку на основе двух C++ header-only файлов:

- `aruco_nano_v6.h` — детектор квадратных маркеров (ARUCO_MIP_36h12, AprilTag 36h11).
- `nanofractal.h` — детектор фрактальных маркеров (FRACTAL_2L_6…5L_6), устойчивых к окклюзиям.

Главный приоритет — **максимальная производительность**: минимальная задержка одиночного кадра (реалтайм) при сохранении возможности быстрой параллельной офлайн-обработки (batch).

## 2. Выбор версии ArUco Nano: v6

Сравнение `aruco_nano_v5.h` и `aruco_nano_v6.h`:

| Критерий | v5 | v6 |
|---|---|---|
| Исходный код | обфусцирован (имена-хеши) | чистый, читаемый |
| Окно `adaptiveThreshold` | 7 | 13 (надёжнее детектит границы) |
| Словари | ARUCO_MIP_36h12 + AprilTag 36h11 | те же |
| API `detect()` | идентичен | идентичен |
| Лицензия | Apache 2.0 | Apache 2.0 |

**Решение: v6.** Читаемый код позволяет оптимизировать горячий путь и поддерживать обёртку; детекция надёжнее; производительность по сути идентична (разница окна адаптивного порога ничтожна). Обфускация v5 не нужна для нашей задачи.

«Два заголовочных файла» = `nanofractal.h` + `aruco_nano_v6.h`.

## 3. Объём

- Экспонируются **оба детектора**: фрактальный и квадратный (ArUco/AprilTag).
- **Оценка позы** (rvec/tvec через `solvePnP` / `SOLVEPNP_IPPE`) включена в API.
- Поставка — **переносимые wheels на PyPI** (cibuildwheel + вендоренный минимальный OpenCV).
- Биндинг — **nanobind**.

## 4. Факты из исходников (определяют архитектуру)

### 4.1 Публичные API заголовков
- **ArUco Nano v6:** `aruconano::MarkerDetector::detect(img, maxAttemptsPerCandidate=10, dict)` — **статический, без состояния**. `Marker : std::vector<cv::Point2f>` с `id`, `estimatePose(camMatrix, distCoeffs, size)`, `draw()`. Словарь из ~256–512 кодов строится **на каждом вызове** (выносимо из горячего цикла).
- **nanofractal:** `nanofractal::FractalMarkerDetector` с `setParams(config, markerSize=-1)`, `detect(img)`, `detect(img, p3d, p2d)`. Конфиг (`FractalMarkerSet`) строится один раз и **дорогой** (парсинг встроенного бинарника + предрасчёт keypoints). Использует `cv::FastFeatureDetector`, KD-дерево (picoflann), гомографию.

### 4.2 Модули OpenCV, реально нужные
`core`, `imgproc`, `calib3d`, `features2d`. **НЕ нужны** `highgui`, `videoio`, `dnn`, `ml`, `photo`, `objdetect`, `gapi` и пр. (встречаются только в примерах в комментариях). Это позволяет собрать минимальный OpenCV → лёгкие wheels.

### 4.3 Потокобезопасность
- `MarkerDetector::detect` (ArUco) — статический, локальное состояние → **потокобезопасен** (один экземпляр можно звать из многих потоков).
- `FractalMarkerDetector::detect` — **не const**: лениво кеширует keypoints и использует `std::map::operator[]` → **НЕ потокобезопасен на одном экземпляре**. Для параллельного batch нужен **пул независимых детекторов** (по одному на рабочий поток).

## 5. Архитектура

```
Python API  (nanofractal/__init__.py + .pyi)
   ArucoDetector · FractalDetector · DetectionResult
        │  nanobind (zero-copy ndarray, GIL release)
Binding layer  (src/_bindings.cpp)
   • numpy → cv::Mat без копии
   • вывод в преаллоцированные numpy-массивы
   • пул детекторов для параллельного batch
Vendored headers (без правок): aruco_nano_v6.h, nanofractal.h
Минимальный статический OpenCV: core/imgproc/calib3d/features2d
```

**Принцип:** заголовки вендорятся **без изменений** (лёгкое обновление с upstream). Вся производительность — в слое биндинга (zero-copy, GIL, переиспользование детектора, параллелизм). Микро-оптимизации внутри заголовков (вынос построения словаря ArUco из цикла, переиспользование FAST-детектора) — **опциональны**, отдельным последующим шагом (единицы процентов, форкают upstream).

### Структура репозитория
```
nanofractal/
├── CMakeLists.txt              # scikit-build-core + CMake
├── pyproject.toml              # сборка, метаданные, cibuildwheel
├── third_party/                # aruco_nano_v6.h, nanofractal.h (вендоренные)
├── src/_bindings.cpp           # nanobind модуль
├── src/nanofractal/__init__.py # тонкий Python-фасад
├── src/nanofractal/__init__.pyi# типы (numpy.typing)
├── tests/                      # pytest: корректность + бенчмарки
└── .github/workflows/wheels.yml# cibuildwheel
```

## 6. Python API

```python
import numpy as np
from nanofractal import ArucoDetector, FractalDetector, Dict

# --- ArUco / AprilTag ---
det = ArucoDetector(dictionary=Dict.ARUCO_MIP_36h12, max_attempts=1)
res = det.detect(frame)                 # frame: np.uint8 (H,W) или (H,W,3) BGR
res.ids                                 # np.int32   (N,)
res.corners                             # np.float32 (N, 4, 2)
rvecs, tvecs = det.estimate_pose(res.corners, camera_matrix, dist_coeffs, marker_size=0.05)
# rvecs, tvecs: np.float64 (N, 3) — батчево по всем маркерам

# --- Fractal ---
fdet = FractalDetector("FRACTAL_5L_6", marker_size=0.85)   # конфиг строится 1 раз
fres = fdet.detect(frame)               # внешние углы + id
fres.ids                                # np.int32   (N,)
fres.corners                            # np.float32 (N, 4, 2)
fres = fdet.detect(frame, with_inner_points=True)
fres.points_2d                          # np.float32 (M, 2)
fres.points_3d                          # np.float32 (M, 3)

# --- Параллельный batch (офлайн) ---
results = det.detect_batch(frames, num_threads=0)   # 0 = все ядра
```

**`max_attempts` (компромисс скорость/устойчивость).** В заголовке дефолт = 10 (до 10 попыток детекции на кандидата со случайным сдвигом углов — медленнее, но устойчивее). Для реалтайма дефолт обёртки = **1** (один проход, без retry-цикла — максимальная скорость). Пользователь поднимает до 10 при необходимости устойчивости. Документируется явно.

### Решения по производительности
1. **Zero-copy вход.** C-contiguous `uint8` ndarray оборачивается в `cv::Mat` поверх того же буфера. Не-contiguous → одна явная копия (документируется).
2. **GIL отпускается** на время `detect`/`estimate_pose` (`nb::gil_scoped_release`).
3. **Вывод в numpy одним проходом** в C++ — без поэлементного создания Python-объектов в горячем пути.
4. **Переиспользование детектора.** Фрактальный конфиг строится раз в конструкторе; ArUco-словарь выносится из цикла и строится раз на детектор.
5. **Параллельный batch через пул.** `detect_batch` держит пул из `num_threads` независимых детекторов (для фрактального — обязательно из-за непотокобезопасности; для ArUco единообразия ради), раздаёт кадры через `parallel_for`.

### Обработка ошибок
- Неверный dtype/форма входа → `TypeError`/`ValueError` (проверка до zero-copy).
- Неверный fractal-конфиг → `ValueError` (проброс `std::runtime_error`).
- Пустой результат → пустые массивы корректной формы (`(0,)`, `(0,4,2)`), не `None`.
- Несовпадение размеров `camera_matrix`/`dist_coeffs` → `ValueError`.

## 7. Сборка и поставка

- **scikit-build-core + CMake**, C++17, `-O3`, **LTO**, базовый SIMD **x86-64-v2 (SSE4.2)** для переносимости (OpenCV внутри делает рантайм-диспетч AVX2/AVX-512).
- nanobind через CMake `FetchContent` / build-зависимость; **stable ABI** (один wheel на много версий Python).
- **OpenCV (вариант A):** в CI собирается минимальный статический OpenCV (только `core/imgproc/calib3d/features2d`, `-DBUILD_LIST=...`, без кодеков/highgui/dnn/тестов/доков), линкуется статически. Размер wheel ~5–15 МБ. Сборка кешируется (ccache + кеш артефакта).
- **cibuildwheel:** приоритет Linux `manylinux_2_28` x86-64 (опц. aarch64); CPython 3.9–3.13; `auditwheel` для проверки переносимости. macOS/Windows — расширение позже по той же схеме.

### Отклонённые альтернативы по OpenCV
- **B. Зависеть от `opencv-python`** — хрупкий ABI между сборкой и рантаймом, тяжёлая неконтролируемая зависимость.
- **C. Системный OpenCV** — не переносимо, противоречит цели «wheels на PyPI».

## 8. Тестирование

1. **Корректность:** синтетический рендер изображений с известными маркерами (id + позиции) → проверка id и углов в субпиксельном допуске. Отдельно для ARUCO_MIP_36h12, AprilTag 36h11, FRACTAL_*.
2. **Эталонное сравнение:** маленькая C++-утилита, вызывающая заголовки напрямую, на тех же картинках; результат Python-обёртки совпадает в пределах float-эпсилона.
3. **Параллельная корректность:** `detect_batch` (пул) против последовательного `detect` — идентичные результаты (ловит гонки в пуле фрактального детектора).
4. **Граничные случаи:** пустой кадр, нет маркеров, не-contiguous вход, серый vs BGR, неверный конфиг.

## 9. Бенчмарки (отдельная сюита, pytest-benchmark)

- **Реалтайм:** медиана/p99 латентности одиночного `detect` на 640×480, 1280×720, 1920×1080.
- **Офлайн:** throughput `detect_batch` (кадров/с) vs число потоков — проверка масштабирования.
- **Контроль zero-copy/GIL:** отсутствие лишних копий (счётчик аллокаций / id буфера) и реальное освобождение GIL (параллельный busy-поток не блокируется).

## 10. Критерии готовности

- `pip install` ставит рабочий wheel на чистом Linux без системного OpenCV.
- Детекция совпадает с эталоном C++.
- Латентность одиночного кадра ≈ нативному C++ (накладные расходы биндинга < нескольких %).
- `detect_batch` масштабируется ~линейно по ядрам.

## 11. Открытые / отложенные вопросы

- Имя пакета: предлагается `nanofractal` (import) с классами `ArucoDetector` / `FractalDetector`. Уточнить при необходимости.
- macOS/Windows wheels — после стабильной Linux-поставки.
- Микро-патчи заголовков (вынос словаря ArUco, переиспользование FAST) — опциональный шаг после базовой версии.
