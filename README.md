# Federated Learning DistilBERT AG_NEWS

README ini hanya berisi cara menjalankan mode synchronous dan asynchronous (immediate atau buffered), serta fungsi masing-masing konfigurasi.

## Prasyarat

```bash
pip install -e .
```

## Konfigurasi Jumlah Client (num-supernodes)

Sebelum menjalankan eksperimen, konfigurasikan jumlah simulated client nodes via CLI Flower:

```bash
flwr federation simulation-config --num-supernodes 100 --client-resources-num-cpus 1 --client-resources-num-gpus 0
```

Parameter:

- `--num-supernodes`: jumlah total client yang tersedia dalam simulasi federasi.
- `--client-resources-num-cpus`: alokasi CPU per client.
- `--client-resources-num-gpus`: alokasi GPU per client (gunakan nilai pecahan untuk berbagi GPU antar client).

Nilai `fraction-train` dan `fraction-evaluate` pada `--run-config` menentukan proporsi dari `num-supernodes` yang disampling setiap round.

## Menjalankan Mode Synchronous

Mode ini memakai:

- `pytorchexample/server_app.py`
- `pytorchexample/client_app.py`

Langkah:

1. Pastikan `tool.flwr.app.components.serverapp` dan `tool.flwr.app.components.clientapp` di `pyproject.toml` mengarah ke file synchronous.
2. Jalankan:

```bash
flwr run . --stream
```

Contoh override config synchronous:

```bash
flwr run . --stream --run-config "num-server-rounds=3 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.1 batch-size=32 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true sync-optimizer='fedavg' min-train-nodes=20 min-evaluate-nodes=40 min-available-nodes=10"
```

Pilihan optimizer sync:

- `sync-optimizer='fedavg'` (default)
- `sync-optimizer='fedadagrad'`

Contoh synchronous dengan straggler simulation aktif:

```bash
flwr run . --stream --run-config "num-server-rounds=3 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.1 batch-size=32 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true sync-optimizer='fedavg' straggler-enabled=true straggler-scenario='balanced' baseline-mode='measured'"
```

## Menjalankan Mode Asynchronous

Mode ini memakai:

- `pytorchexample/server_app_async.py`
- `pytorchexample/client_app_async.py`

Langkah:

1. Ubah `tool.flwr.app.components.serverapp` ke `pytorchexample.server_app_async:app`.
2. Ubah `tool.flwr.app.components.clientapp` ke `pytorchexample.client_app_async:app`.
3. Jalankan:

```bash
flwr run . --stream
```

Mode asynchronous dipilih dengan `async-strategy`:

- `async-strategy="immediate"`: update global model setiap ada 1 reply client.
- `async-strategy="buffered"`: kumpulkan reply sampai `async-buffer-size`, lalu update global model dengan FedAvg.

Contoh override config asynchronous immediate:

```bash
flwr run . --stream --run-config "num-server-rounds=3 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.1 batch-size=32 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 async-max-in-flight=4 async-evaluate-interval=4 async-strategy='immediate' straggler-enabled=true straggler-scenario='balanced' baseline-mode='measured'"
```

Contoh override config asynchronous buffered:

```bash
flwr run . --stream --run-config "num-server-rounds=3 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.1 batch-size=32 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 async-max-in-flight=4 async-evaluate-interval=4 async-strategy='buffered' async-buffer-size=4 straggler-enabled=true straggler-scenario='slow_dominant' baseline-mode='fixed' baseline-time-seconds=60.0"
```

## Penjelasan Konfigurasi

Konfigurasi aktif dibaca dari `tool.flwr.app.config` pada `pyproject.toml`.

### Preset Eksekusi: Laptop vs Super VM

Untuk memudahkan, gunakan dua preset berikut.

| Preset        | Tujuan                                         | Rekomendasi Kunci                                                                            |
| ------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `Laptop-safe` | Stabil dulu di perangkat lokal                 | `async-max-in-flight=2`, `batch-size=16`, `min-*-nodes=2`, `straggler-enabled=false`         |
| `Super-VM`    | Throughput tinggi saat resource besar tersedia | `async-max-in-flight=8..16`, `batch-size=32..64`, `min-*-nodes>=8`, `straggler-enabled=true` |

Default `pyproject.toml` saat ini sudah di-set ke preset `Laptop-safe`.

Contoh override untuk preset `Super-VM` (nanti saat VM siap):

```bash
flwr run . --stream --run-config "num-server-rounds=12 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.02 batch-size=32 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true min-train-nodes=8 min-evaluate-nodes=8 min-available-nodes=8 async-strategy='immediate' async-max-in-flight=8 async-evaluate-interval=1 straggler-enabled=true straggler-scenario='balanced' baseline-mode='measured' train-timeout-seconds=600.0"
```

Arti masing-masing parameter:

- `num-server-rounds`: jumlah round global (atau target round logis pada async).
- `stop-mode`: `num_rounds` (default) atau `target_accuracy` untuk berhenti saat akurasi tercapai.
- `target-accuracy`: ambang Top-1 Test Accuracy untuk menghentikan training ketika `stop-mode=target_accuracy`.
- `fraction-train`: proporsi client yang disampling untuk training.
- `fraction-evaluate`: proporsi client yang disampling untuk evaluasi client-side.
- `local-epochs`: jumlah epoch training lokal per client terpilih.
- `learning-rate`: learning rate optimizer untuk local training.
- `batch-size`: ukuran batch pada DataLoader client.
- `weight-decay`: regularisasi AdamW.
- `warmup-ratio`: rasio warmup scheduler linear.
- `max-grad-norm`: nilai clip gradien maksimum.
- `use-lora`: aktif/nonaktif LoRA (default: `true`).
- `sync-optimizer`: optimizer server sync (`fedavg` atau `fedadagrad`).
- `train-timeout-seconds`: timeout idle loop async saat tidak ada reply baru.
- `reply-poll-interval-seconds`: interval polling inbox reply server async.
- `async-max-in-flight`: jumlah maksimum request training yang berjalan paralel di async.
- `async-evaluate-interval`: evaluasi global dilakukan setiap N update async.
- `async-strategy`: pilih strategi asynchronous (`immediate` atau `buffered`).
- `async-buffer-size`: jumlah reply client yang dikumpulkan sebelum dilakukan update model global (khusus mode asynchronous buffered).
- `straggler-enabled`: aktif/nonaktif simulasi straggler pada client train.
- `straggler-scenario`: distribusi tier straggler (`balanced`, `slow_dominant`, `fast_dominant`).
- `baseline-mode`: mode baseline waktu (`measured` memakai train time lokal client, `fixed` memakai nilai konstan).
- `baseline-time-seconds`: baseline waktu konstan saat `baseline-mode='fixed'`.
- `staleness-weighting-enabled`: aktif/nonaktif staleness-aware weighting pada mode asynchronous.
- `staleness-weighting-mode`: mode weighting (`polynomial`, `fedstaleweight`, atau alias `fair`).
- `staleness-exponent`: eksponen alpha untuk mode `polynomial`.
- `fedstaleweight-ema-beta`: faktor EMA untuk estimasi expected staleness pada mode `fedstaleweight`.

## Metrik yang Dilog ke W&B

- `top1_test_accuracy`: akurasi Top-1 pada testing set.
- `direction_variation`: variasi arah LoRA antar pembaruan global.
- `number_of_client_trips_to_target_accuracy`: jumlah client trips sampai target accuracy tercapai.
- `wall_clock_time_to_target_accuracy`: waktu wall-clock sampai target accuracy tercapai.
- `vram_allocated_mb`: VRAM terpakai pada client (MB).
- `vram_reserved_mb`: VRAM reserved pada client (MB).
- `communication_bytes`: ukuran payload update per client (byte).
- `communication_megabytes`: ukuran payload update per client (MB).
- `communication_params`: jumlah parameter update per client.
- `relative_bandwidth_ratio`: rasio ukuran LoRA terhadap full fine-tuning ($\text{LoRA bytes} / \text{full bytes}$).

## Tabel Skenario Straggler (FAST/MEDIUM/SLOW)

Tier straggler ditentukan deterministik dari `partition_id` dan menggunakan multiplier berikut:

| Tier   | Multiplier Waktu Target | Interpretasi                                     |
| ------ | ----------------------- | ------------------------------------------------ |
| FAST   | x1.0                    | Klien cepat, target selesai sekitar baseline `T` |
| MEDIUM | x1.5                    | Klien menengah, target selesai sekitar `1.5T`    |
| SLOW   | x3.0                    | Klien lambat, target selesai sekitar `3.0T`      |

Distribusi tier per nilai `straggler-scenario`:

| `straggler-scenario` | FAST | MEDIUM | SLOW       |
| -------------------- | ---- | ------ | ---------- |
| `balanced`           | 33%  | 33%    | 34% (sisa) |
| `slow_dominant`      | 20%  | 10%    | 70% (sisa) |
| `fast_dominant`      | 70%  | 20%    | 10% (sisa) |

Catatan:

- Jumlah aktual klien per tier dihitung dengan pembulatan (`round`) untuk FAST dan MEDIUM, sedangkan SLOW mengambil sisa klien.
- Untuk 100 klien, kira-kira menjadi: `balanced=33/33/34`, `slow_dominant=20/10/70`, `fast_dominant=70/20/10`.

## Staleness-Aware Weighting (Async Only)

Ketika `staleness-weighting-enabled=true`, bobot update client dihitung berdasarkan mode yang dipilih:

- `staleness-weighting-mode='polynomial'`: penalti staleness klasik.
- `staleness-weighting-mode='fedstaleweight'` (atau `fair`): fairness boost berbasis expected staleness (EMA).

Formula mode `polynomial`:

```
tau               = updates_done_at_receive - updates_done_at_dispatch
staleness_weight  = (tau + 1)^(-alpha)
final_weight      = num_examples * staleness_weight
```

- `tau = 0` artinya reply tiba sebelum ada update lain terjadi → bobot penuh.
- `tau > 0` artinya ada `tau` update lain yang sudah diintegrasikan sejak dispatch → bobot berkurang.
- `alpha` diatur lewat `staleness-exponent`. Nilai lebih besar = penalti lebih besar terhadap update yang basi.

Pada mode immediate, `tau` dan `staleness_weight` dilog per-update ke Flower log dan W&B.
Pada mode buffered, rata-rata `avg_expected_staleness` dan `avg_fairness_boost` dari seluruh buffer dilog ke W&B per agregasi.

Contoh override config asynchronous immediate dengan staleness weighting:

```bash
flwr run . --stream --run-config "num-server-rounds=3 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.1 batch-size=32 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 async-max-in-flight=4 async-evaluate-interval=4 async-strategy='immediate' staleness-weighting-enabled=true staleness-exponent=0.5 straggler-enabled=true straggler-scenario='balanced' baseline-mode='measured'"
```

Contoh override config asynchronous buffered dengan staleness weighting:

```bash
flwr run . --stream --run-config "num-server-rounds=3 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.1 batch-size=32 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 train-timeout-seconds=600.0 reply-poll-interval-seconds=2.0 async-max-in-flight=4 async-evaluate-interval=4 async-strategy='buffered' async-buffer-size=4 staleness-weighting-enabled=true staleness-exponent=0.5 straggler-enabled=true straggler-scenario='slow_dominant' baseline-mode='fixed' baseline-time-seconds=60.0"
```

## Config Template untuk Perbandingan Fair: Immediate vs Buffered

Untuk membandingkan step-0 accuracy dengan **config identik** (hanya strategy dan weighting mode berbeda), gunakan template berikut.

### Template 1: Async Immediate (Polynomial Staleness Weighting)

**File: `pyproject.toml` atau CLI override**

```toml
[tool.flwr.app.config]
num-server-rounds = 5
stop-mode = "num_rounds"
target-accuracy = 0.90
fraction-train = 0.005
fraction-evaluate = 0.01
local-epochs = 1
learning-rate = 0.003
lr-decay-interval = 20
lr-decay-factor = 0.9
min-learning-rate = 0.0001
min-train-nodes = 1
min-evaluate-nodes = 1
min-available-nodes = 1
batch-size = 8
weight-decay = 0.01
warmup-ratio = 0.1
max-grad-norm = 1.0
use-lora = true
train-timeout-seconds = 180.0
reply-poll-interval-seconds = 2.0
async-max-in-flight = 1
async-evaluate-interval = 1
async-strategy = "immediate"
async-buffer-size = 4
straggler-enabled = false
straggler-scenario = "balanced"
baseline-mode = "fixed"
baseline-time-seconds = 30.0
staleness-weighting-enabled = false
staleness-weighting-mode = "polynomial"
staleness-exponent = 0.5
fedstaleweight-ema-beta = 0.8
```

**CLI command:**

```bash
flwr run . --stream --run-config "num-server-rounds=5 stop-mode=num_rounds target-accuracy=0.90 fraction-train=0.005 fraction-evaluate=0.01 local-epochs=1 learning-rate=0.003 lr-decay-interval=20 lr-decay-factor=0.9 min-learning-rate=0.0001 min-train-nodes=1 min-evaluate-nodes=1 min-available-nodes=1 batch-size=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true train-timeout-seconds=180.0 reply-poll-interval-seconds=2.0 async-max-in-flight=1 async-evaluate-interval=1 async-strategy='immediate' async-buffer-size=4 straggler-enabled=false straggler-scenario='balanced' baseline-mode='fixed' baseline-time-seconds=30.0 staleness-weighting-enabled=false staleness-weighting-mode='polynomial' staleness-exponent=0.5 fedstaleweight-ema-beta=0.8"
```

### Template 2: Async Buffered (FedStaleWeight Fairness Weighting)

**File: `pyproject.toml` atau CLI override**

```toml
[tool.flwr.app.config]
num-server-rounds = 5
stop-mode = "num_rounds"
target-accuracy = 0.90
fraction-train = 0.005
fraction-evaluate = 0.01
local-epochs = 1
learning-rate = 0.003
lr-decay-interval = 20
lr-decay-factor = 0.9
min-learning-rate = 0.0001
min-train-nodes = 1
min-evaluate-nodes = 1
min-available-nodes = 1
batch-size = 8
weight-decay = 0.01
warmup-ratio = 0.1
max-grad-norm = 1.0
use-lora = true
train-timeout-seconds = 180.0
reply-poll-interval-seconds = 2.0
async-max-in-flight = 1
async-evaluate-interval = 1
async-strategy = "buffered"
async-buffer-size = 4
straggler-enabled = false
straggler-scenario = "balanced"
baseline-mode = "fixed"
baseline-time-seconds = 30.0
staleness-weighting-enabled = false
staleness-weighting-mode = "fedstaleweight"
staleness-exponent = 0.5
fedstaleweight-ema-beta = 0.8
```

**CLI command:**

````bash
flwr run . --stream --run-config "num-server-rounds=5 stop-mode=num_rounds target-accuracy=0.90 fraction-train=0.005 fraction-evaluate=0.01 local-epochs=1 learning-rate=0.003 lr-decay-interval=20 lr-decay-factor=0.9 min-learning-rate=0.0001 min-train-nodes=1 min-evaluate-nodes=1 min-available-nodes=1 batch-size=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true train-timeout-seconds=180.0 reply-poll-interval-seconds=2.0 async-max-in-flight=1 async-evaluate-interval=1 async-strategy='buffered' async-buffer-size=4 straggler-enabled=false straggler-scenario='balanced' baseline-mode='fixed' baseline-time-seconds=30.0 staleness-weighting-enabled=false staleness-weighting-mode='fedstaleweight' staleness-exponent=0.5 fedstaleweight-ema-beta=0.8"

## Template Stop Berdasarkan Target Accuracy

Gunakan `stop-mode="target_accuracy"` untuk menghentikan training saat Top-1 Test Accuracy mencapai ambang.

### Template 3: Async Immediate (Stop by Target Accuracy)

**File: `pyproject.toml` atau CLI override**

```toml
[tool.flwr.app.config]
num-server-rounds = 50
stop-mode = "target_accuracy"
target-accuracy = 0.90
fraction-train = 0.005
fraction-evaluate = 0.01
local-epochs = 1
learning-rate = 0.003
lr-decay-interval = 20
lr-decay-factor = 0.9
min-learning-rate = 0.0001
min-train-nodes = 1
min-evaluate-nodes = 1
min-available-nodes = 1
batch-size = 8
weight-decay = 0.01
warmup-ratio = 0.1
max-grad-norm = 1.0
use-lora = true
train-timeout-seconds = 180.0
reply-poll-interval-seconds = 2.0
async-max-in-flight = 1
async-evaluate-interval = 1
async-strategy = "immediate"
async-buffer-size = 4
straggler-enabled = false
straggler-scenario = "balanced"
baseline-mode = "fixed"
baseline-time-seconds = 30.0
staleness-weighting-enabled = false
staleness-weighting-mode = "polynomial"
staleness-exponent = 0.5
fedstaleweight-ema-beta = 0.8
````

**CLI command:**

```bash
flwr run . --stream --run-config "num-server-rounds=50 stop-mode=target_accuracy target-accuracy=0.90 fraction-train=0.005 fraction-evaluate=0.01 local-epochs=1 learning-rate=0.003 lr-decay-interval=20 lr-decay-factor=0.9 min-learning-rate=0.0001 min-train-nodes=1 min-evaluate-nodes=1 min-available-nodes=1 batch-size=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true train-timeout-seconds=180.0 reply-poll-interval-seconds=2.0 async-max-in-flight=1 async-evaluate-interval=1 async-strategy='immediate' async-buffer-size=4 straggler-enabled=false straggler-scenario='balanced' baseline-mode='fixed' baseline-time-seconds=30.0 staleness-weighting-enabled=false staleness-weighting-mode='polynomial' staleness-exponent=0.5 fedstaleweight-ema-beta=0.8"
```

### Template 4: Async Buffered (Stop by Target Accuracy)

**File: `pyproject.toml` atau CLI override**

```toml
[tool.flwr.app.config]
num-server-rounds = 50
stop-mode = "target_accuracy"
target-accuracy = 0.90
fraction-train = 0.005
fraction-evaluate = 0.01
local-epochs = 1
learning-rate = 0.003
lr-decay-interval = 20
lr-decay-factor = 0.9
min-learning-rate = 0.0001
min-train-nodes = 1
min-evaluate-nodes = 1
min-available-nodes = 1
batch-size = 8
weight-decay = 0.01
warmup-ratio = 0.1
max-grad-norm = 1.0
use-lora = true
train-timeout-seconds = 180.0
reply-poll-interval-seconds = 2.0
async-max-in-flight = 1
async-evaluate-interval = 1
async-strategy = "buffered"
async-buffer-size = 4
straggler-enabled = false
straggler-scenario = "balanced"
baseline-mode = "fixed"
baseline-time-seconds = 30.0
staleness-weighting-enabled = false
staleness-weighting-mode = "fedstaleweight"
staleness-exponent = 0.5
fedstaleweight-ema-beta = 0.8
```

**CLI command:**

```bash
flwr run . --stream --run-config "num-server-rounds=50 stop-mode=target_accuracy target-accuracy=0.90 fraction-train=0.005 fraction-evaluate=0.01 local-epochs=1 learning-rate=0.003 lr-decay-interval=20 lr-decay-factor=0.9 min-learning-rate=0.0001 min-train-nodes=1 min-evaluate-nodes=1 min-available-nodes=1 batch-size=8 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 use-lora=true train-timeout-seconds=180.0 reply-poll-interval-seconds=2.0 async-max-in-flight=1 async-evaluate-interval=1 async-strategy='buffered' async-buffer-size=4 straggler-enabled=false straggler-scenario='balanced' baseline-mode='fixed' baseline-time-seconds=30.0 staleness-weighting-enabled=false staleness-weighting-mode='fedstaleweight' staleness-exponent=0.5 fedstaleweight-ema-beta=0.8"
```

```

### Perbedaan Kunci

| Parameter                  | Immediate (Poly) | Buffered (FedStaleWeight) |
| -------------------------- | ---------------- | ------------------------- |
| `async-strategy`           | `"immediate"`    | `"buffered"`              |
| `staleness-weighting-mode` | `"polynomial"`   | `"fedstaleweight"`        |
| **Semua config lainnya**   | **Identik**      | **Identik**               |

**Catatan:**

- Kedua template memiliki config identik kecuali `async-strategy` dan `staleness-weighting-mode`.
- Jalankan kedua mode dengan config identik pada **environment yang sama** (local atau VM) untuk perbandingan fair.
- Step-0 accuracy seharusnya **identik atau sangat dekat** jika tidak ada update yang diterapkan sebelum evaluasi awal.
```
