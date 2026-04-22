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
flwr run . --stream --run-config "num-server-rounds=3 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.1 batch-size=32 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 min-train-nodes=20 min-evaluate-nodes=40 min-available-nodes=10"
```

Contoh synchronous dengan straggler simulation aktif:

```bash
flwr run . --stream --run-config "num-server-rounds=3 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.1 batch-size=32 straggler-enabled=true straggler-scenario='balanced' baseline-mode='measured'"
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
flwr run . --stream --run-config "num-server-rounds=3 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.1 batch-size=32 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 train-timeout-seconds=120.0 reply-poll-interval-seconds=2.0 async-max-in-flight=4 async-evaluate-interval=4 async-strategy='immediate' straggler-enabled=true straggler-scenario='balanced' baseline-mode='measured'"
```

Contoh override config asynchronous buffered:

```bash
flwr run . --stream --run-config "num-server-rounds=3 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.1 batch-size=32 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 train-timeout-seconds=120.0 reply-poll-interval-seconds=2.0 async-max-in-flight=4 async-evaluate-interval=4 async-strategy='buffered' async-buffer-size=4 straggler-enabled=true straggler-scenario='slow_dominant' baseline-mode='fixed' baseline-time-seconds=60.0"
```

## Penjelasan Konfigurasi

Konfigurasi aktif dibaca dari `tool.flwr.app.config` pada `pyproject.toml`.
Preset referensi ada di:

- `tool.pytorchexample.config.synchronous`
- `tool.pytorchexample.config.asynchronous`
- `tool.pytorchexample.config.asynchronous-buffered`

Arti masing-masing parameter:

- `num-server-rounds`: jumlah round global (atau target round logis pada async).
- `fraction-train`: proporsi client yang disampling untuk training.
- `fraction-evaluate`: proporsi client yang disampling untuk evaluasi client-side.
- `local-epochs`: jumlah epoch training lokal per client terpilih.
- `learning-rate`: learning rate optimizer untuk local training.
- `batch-size`: ukuran batch pada DataLoader client.
- `weight-decay`: regularisasi AdamW.
- `warmup-ratio`: rasio warmup scheduler linear.
- `max-grad-norm`: nilai clip gradien maksimum.
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
- `staleness-weighting-enabled`: aktif/nonaktif polynomial staleness weighting pada mode asynchronous (default `false`).
- `staleness-exponent`: eksponen alpha pada formula staleness weight (default `1.0`).

## Staleness-Aware Weighting (Async Only)

Ketika `staleness-weighting-enabled=true`, setiap update client diberi bobot yang lebih kecil jika reply-nya sudah "basi" (terlambat). Formula:

```
tau               = updates_done_at_receive - updates_done_at_dispatch
staleness_weight  = (tau + 1)^(-alpha)
final_weight      = num_examples * staleness_weight
```

- `tau = 0` artinya reply tiba sebelum ada update lain terjadi → bobot penuh.
- `tau > 0` artinya ada `tau` update lain yang sudah diintegrasikan sejak dispatch → bobot berkurang.
- `alpha` diatur lewat `staleness-exponent`. Nilai lebih besar = penalti lebih besar terhadap update yang basi.

Pada mode immediate, `tau` dan `staleness_weight` dilog per-update ke Flower log dan W&B.
Pada mode buffered, rata-rata `avg_tau` dan `avg_staleness_weight` dari seluruh buffer dilog ke W&B per agregasi.

Contoh override config asynchronous immediate dengan staleness weighting:

```bash
flwr run . --stream --run-config "num-server-rounds=3 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.1 batch-size=32 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 train-timeout-seconds=120.0 reply-poll-interval-seconds=2.0 async-max-in-flight=4 async-evaluate-interval=4 async-strategy='immediate' staleness-weighting-enabled=true staleness-exponent=0.5 straggler-enabled=true straggler-scenario='balanced' baseline-mode='measured'"
```

Contoh override config asynchronous buffered dengan staleness weighting:

```bash
flwr run . --stream --run-config "num-server-rounds=3 fraction-train=0.025 fraction-evaluate=0.05 local-epochs=1 learning-rate=0.1 batch-size=32 weight-decay=0.01 warmup-ratio=0.1 max-grad-norm=1.0 train-timeout-seconds=120.0 reply-poll-interval-seconds=2.0 async-max-in-flight=4 async-evaluate-interval=4 async-strategy='buffered' async-buffer-size=4 staleness-weighting-enabled=true staleness-exponent=0.5 straggler-enabled=true straggler-scenario='slow_dominant' baseline-mode='fixed' baseline-time-seconds=60.0"
```
