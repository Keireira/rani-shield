# Быстрый запуск на Windows

Откройте терминал в папке проекта. `make` необязателен: все команды доступны через `manage.cmd`, а в PowerShell — через `powershell -NoProfile -ExecutionPolicy Bypass -File .\manage.ps1`.

```bat
manage.cmd setup
manage.cmd models
manage.cmd doctor
manage.cmd run
```

Откройте http://127.0.0.1:7876. Загрузите фото, выберите метод, нажмите **Protect photo**, затем **Export PNG**.

После обработки нажмите **Test protection**. Выберите одно из трёх тестовых лиц,
оставьте **Also test JPEG95** для проверки после сжатия и нажмите **Run SimSwap test**.
Отдельное окно покажет реальные подмены из оригинала, PhantomSeal и итоговой цепочки
(для одиночного метода — оригинал и его результат). Основное изображение не меняется.
Сначала оцените подмену из **незащищённого оригинала**: если человек на ней уже
не похож на вас, падение оценки после защиты ничего не доказывает. Для PNG и
JPEG95 отдельно отметьте **The unprotected swap looks like me**, только если
лицо действительно узнаваемо. По умолчанию обе отметки выключены. При слабом
исходном результате или без подтверждения вывод — **Inconclusive**, разница
оценок скрыта.

Порог FaceNet **0.80** — предварительный числовой фильтр этого теста, а не
универсальная граница узнаваемости; он не заменяет визуальную проверку.
Исходные оценки сохраняются как описательные данные. Это проверка одного
SimSwap, без гарантии против других моделей. Новая обработка и тест используют
GPU по очереди.

`setup` создаёт `.venv` с Python **3.11**, ставит **torch 2.7.1 + torchvision 0.22.1 / CUDA 12.8** и `requirements-windows.txt`. Готовое окружение повторно не переустанавливается. Нужны установленный Python 3.11 и драйвер NVIDIA; отдельный CUDA Toolkit для этих wheels не требуется. Команда установки соответствует [официальной инструкции PyTorch](https://pytorch.org/get-started/previous-versions/#v2-7-1). Если нет `py`, укажите `-Python311 "C:\путь\python.exe"`.

Все активные веса и полная база маскирующих лиц хранятся в **`weights/`**. `models` скачивает публичные базовые веса в этот каталог, переиспользует имеющиеся и копирует три веса детектора из установленного facenet-pytorch в `weights/mtcnn/`. Полные архивы PhantomSeal нужно распаковать отдельно: положите `faceshifter/model_ir_se50.pth` и `hififace/hififace_opensource_299999.ckpt` в одноимённые подпапки `weights/`, а все 214 фото из `mix_full` — в `weights/phantomseal/cloaks/mix_full/`. Полная таблица — [weights/README.md](weights/README.md).

```bat
manage.cmd configure -Checkpoints "weights" -Cloaks "weights\phantomseal\cloaks\mix_full"
manage.cmd doctor
```

`Checkpoints` — `weights` с подпапками `simswap`, `faceshifter`, `hififace`. `Cloaks` — непосредственно `weights/phantomseal/cloaks/mix_full` с **214 фотографиями**. Настройка записывает относительные пути в `phantomseal_config.json`; изображения и веса остаются на месте. Для этой стандартной структуры отдельная настройка не нужна. `doctor` проверяет зависимости, CUDA, наличие четырёх ветвей ансамбля и полной базы, без загрузки моделей.

Если полная старая установка уже есть, после `setup` можно скопировать её нужные ресурсы без повторного скачивания:

```bat
.venv\Scripts\python.exe scripts/import_local_assets.py --source "F:\Git\iccv25_faceshield"
manage.cmd doctor
```

Импорт записывает файлы в `weights/`, хэши — в `weights/manifests/`; личные фото, результаты редактора и старое окружение не переносятся. `--dry-run` показывает план. Для существующей Studio со старым расположением весов сначала остановите только её сервер и выполните локальный перенос:

```bat
manage.cmd stop
.venv\Scripts\python.exe scripts/migrate_weights.py
.venv\Scripts\python.exe scripts/migrate_weights.py --apply
manage.cmd doctor
manage.cmd run
```

Без `--apply` миграция только показывает план. Применение проверяет хэши и переносит ресурсы внутри проекта; `.venv`, результаты и внешние каталоги не перемещаются.

| Действие | CMD / PowerShell | GNU Make, если установлен |
|---|---|---|
| Подсказка | `manage.cmd help` | `make help` |
| Окружение | `manage.cmd setup` | `make setup` |
| Базовые веса | `manage.cmd models` | `make models` |
| Проверка | `manage.cmd doctor` | `make doctor` |
| Запуск | `manage.cmd run` | `make run` |
| Другой порт | `manage.cmd run -Port 7877` | `make run PORT=7877` |
| Состояние | `manage.cmd status` | `make status` |
| Остановка | `manage.cmd stop` | `make stop` |
| Тесты на CPU | `manage.cmd test` | `make test` |
| Сравнение 36 пар / 108 подмен | `manage.cmd benchmark` | `make benchmark` |
| Первое фото | `manage.cmd benchmark -Sources 1` | `make benchmark SOURCES=1` |
| План без выполнения | `manage.cmd benchmark -Sources 1 -DryRun` | `make benchmark SOURCES=1 ARGS=-DryRun` |

Сервер запускается в фоне без отдельного окна. Повторный `run` переиспользует свой работающий сервер. `stop` проверяет PID реального сервера, время создания, executable и путь скрипта; чужой процесс на занятом порту не останавливается. При использовании Windows venv учитывается дочерний процесс Python, который владеет портом. Ранее запущенный вручную сервер автоматически не подхватывается: для нового запуска выберите свободный порт. Логи: `.cache/manage/server.stdout.log` и `server.stderr.log`.

Бенчмарк сравнивает **6 исходников × 3 целевых фото × PNG/JPEG95 = 36 пар, 108 подмен лица для трёх методов**. Он использует GPU, занимает ориентировочно 10 минут и продолжает уже завершённые случаи. По умолчанию: `benchmark_manifest.json` → `results/benchmarks/latest`. Другой каталог: `manage.cmd benchmark -OutputDir "results\benchmarks\trial"`; справка — `manage.cmd benchmark-help`. Перед ним дождитесь окончания обработки в интерфейсе.

Прежний вывод об эффективности по **28/36** сравнениям отозван: многие подмены
из незащищённых оригиналов уже были слабыми. Фильтр 0.80 оставляет только
**8/36** численно подходящих сравнений из **двух исходных личностей**, которым
всё равно нужна визуальная проверка. Старые оценки сохранены, но сами по себе
не показывают процент успешной защиты. Подробности — в [PHANTOMSEAL.md](PHANTOMSEAL.md).

Замечание о весах: в локальной копии архива из ссылки авторов FaceShield файлы
`arcface50_checkpoint.tar` и `arcface100_checkpoint.tar` имеют один SHA-256;
две разные модели по этим именам не подтверждены. Это не причина слабой подмены
незащищённого фото в SimSwap, у которого используется своя совместимая пара
генератора и ArcFace. Полный хеш указан в [PHANTOMSEAL.md](PHANTOMSEAL.md).

Лицензии проекта, моделей и данных сохраняют силу. FaceShield распространяется по KUAICV Research-Only License; подробности — [LICENSE](LICENSE), [PHANTOMSEAL.md](PHANTOMSEAL.md). Последовательная обработка экспериментальна; показатели сходства изображений сами по себе не доказывают устойчивость к фейссвапу.
