# Dia Final Project

Model: `nari-labs/Dia-1.6B-0626`

External mapping:

- `[S1]` = agent
- `[S2]` = customer

## Included

- `server.py` — same server, same port, two endpoints:
  - `POST /tts` production dual-reference TTS
  - `POST /tts_seed` seed auditioning
- `client.py` — supports both endpoints automatically
- `extract_voices.py` — local utility to extract `agent.wav` and many customer options from two-speaker seed WAVs
- `Dockerfile`
- `requirements.txt`
- `client_requirements.txt`
- `full_call.txt`
- `reference/seed_reference.txt`

## Long-text behavior

The production endpoint never sends the whole long conversation through one `model.generate()`.

It parses the script in order:

```text
[S1] agent turn
[S2] customer turn
[S1] next agent turn
[S2] next customer turn
```

and generates:

```text
chunk 1 = S1 agent turn
chunk 2 = S2 customer turn
chunk 3 = S1 next agent turn
chunk 4 = S2 next customer turn
```

If one turn is longer than `MAX_WORDS_PER_CHUNK` (default `18`), that individual turn is split further at sentence boundaries.

Every returned chunk is appended in transcript order into one final WAV. Playback is also sequential: chunk N+1 never starts playing until chunk N finishes.

## Why the internal reference order changes

Dia recommends alternating `[S1]` / `[S2]` speaker tags. For independent agent/customer reference WAVs, the server uses:

Agent chunk:

```text
reference audio = agent.wav + customer.wav
prompt          = [S1] agent-reference [S2] customer-reference [S1] target-agent-text
```

Customer chunk:

```text
reference audio = customer.wav + agent.wav
prompt          = [S1] customer-reference [S2] agent-reference [S1] target-customer-text
```

Externally, your transcript still remains `[S1]=agent`, `[S2]=customer`.

## Server build

```bash
docker build -t dia-final .
```

Run:

```bash
docker run --rm \
  --gpus all \
  --ipc=host \
  --shm-size=8g \
  -p 8000:8000 \
  --name dia-final \
  dia-final
```

Background:

```bash
docker run -d \
  --gpus all \
  --ipc=host \
  --shm-size=8g \
  -p 8000:8000 \
  --name dia-final \
  dia-final
```

Logs:

```bash
docker logs -f dia-final
```

Health:

```bash
curl http://localhost:8000/health
```

## Local client dependencies

```powershell
python -m pip install -r client_requirements.txt
```

## Seed auditioning

Single seed:

```powershell
python client.py --server http://localhost:8000 --seed 15423 --text "[S1] Hello. Thank you for calling Inspira Financial. How can I help you today? [S2] Hello. I need assistance with my account. Could you please help me with that?" --output seed_15423.wav
```

100 random unique seeds:

```powershell
$seeds=@(); while($seeds.Count -lt 100){$seeds+=Get-Random -Minimum 100 -Maximum 50000;$seeds=@($seeds|Sort-Object -Unique)}; foreach($seed in $seeds){Write-Host "Testing seed: $seed"; python client.py --server http://localhost:8000 --seed $seed --text "[S1] Hello. Thank you for calling Inspira Financial. How can I help you today? [S2] Hello. I need assistance with my account. Could you please help me with that?" --output "seed_$seed.wav"}
```

## Extract agent and customer options

```powershell
python extract_voices.py --input-dir shortlisted\random_wavs --reference-text reference\seed_reference.txt --agent-file seed_15423.wav --output-dir extracted_voices
```

Outputs:

```text
extracted_voices/
├── agent.wav
├── agent.txt
├── customer.txt
└── customers/
    ├── customer_seed_121.wav
    ├── customer_seed_15423.wav
    └── ...
```

## Production with direct text

```powershell
python client.py `
  --server http://localhost:8000 `
  --text "[S1] Hello. How can I help you today? [S2] I need help with my account." `
  --agent-audio extracted_voices\agent.wav `
  --customer-audio extracted_voices\customers\customer_seed_15423.wav `
  --agent-text-file extracted_voices\agent.txt `
  --customer-text-file extracted_voices\customer.txt `
  --max-new-tokens 3072 `
  --output direct_test.wav `
  --play
```

## Production with long text file

```powershell
python client.py `
  --server http://localhost:8000 `
  --text-file full_call.txt `
  --agent-audio extracted_voices\agent.wav `
  --customer-audio extracted_voices\customers\customer_seed_15423.wav `
  --agent-text-file extracted_voices\agent.txt `
  --customer-text-file extracted_voices\customer.txt `
  --max-new-tokens 3072 `
  --output full_call.wav `
  --play
```

## Speed

Normal:

```powershell
--speed 1.0
```

10% slower:

```powershell
--speed 0.9
```

Save adjusted output too:

```powershell
--speed 0.9 --save-adjusted
```

## If text is still skipped

Reduce the chunk size further when starting the container:

```bash
docker run ... -e MAX_WORDS_PER_CHUNK=12 ...
```

Dia is generative, so smaller chunks reduce omissions/repetitions substantially but cannot formally guarantee exactly-once pronunciation of every word.
