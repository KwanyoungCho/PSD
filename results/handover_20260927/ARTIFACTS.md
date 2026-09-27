# Git과 원시 데이터 이관

인계는 **Git 코드/문서/핵심 결과 + 별도 원시 데이터 archive** 두 부분으로 구성된다.
원시 파일을 삭제하거나 요약본으로 대체하지 않았다. 원래 경로와 파일별 SHA256을
[`artifact_manifest.json`](artifact_manifest.json)에 기록했다. 이 manifest의 대상은
아래 일곱 연구 디렉터리다. Engine/루트 문서/사용자 논문은 Git commit이 보존한다.

## Git에 포함한 것

- Production 변경 10개 기존 파일, 신규 probe 3개와 entropy test, 기존 연구 source,
  사용자 KCC2026 PDF, 새 인계 문서/도구 및 진행 기록.
- 일곱 연구 디렉터리의 코드·보고서·수식·계획·frozen calibration·figure·요약 JSON/CSV.
- 기존 full AL의 여섯 run과 posthoc 두 run의 `records.jsonl` 등 핵심 생성 기록.
- Command/env, dataset 질문 snapshot, split/hash, 완료/실패 상태, 작은 진단 결과 및 로그.
  기존 `.gitignore`가 무시하던 일부 연구 로그도 명시적으로 포함했다.
- 9/23 실패 smoke의 command/process/completion/run.log 및 9/27 STOP/상태 문서.

일곱 연구 디렉터리의 Git 대상은 **1,056파일 / 101,869,781 bytes(약97.15MiB)**다.
전체 commit은 여기에 production 코드, 논문, 새 인계 자료가 추가된다.
파일 목록은 [`git_files.txt`](git_files.txt), 체크섬은 [`git_SHA256SUMS`](git_SHA256SUMS).

## 별도 archive에 포함한 것

모든 NPZ, calibration raw timing profile JSON, 10MiB 초과의 tree JSONL이다.
총 **2,362파일 / 23,480,536,914 bytes(약23.48GB, 21.87GiB)**의 파일 내용이다.
NPZ 대부분이 이미 압축돼 있어 별도 tar는 압축하지 않았다. Tar header 때문에
archive 자체 크기는 조금 더 크며 정확한 값/hash는
[`archive_summary.json`](archive_summary.json)에 기록한다.

| 연구 디렉터리 | Git 파일 수 / MiB | Archive 파일 수 / MiB |
|---|---:|---:|
| `results/residial_dist` | 263 / 11.91 | 67 / 225.07 |
| `results/duet_calibration` | 297 / 10.29 | 67 / 956.52 |
| `results/duet_tree_analysis` | 83 / 3.64 | 546 / 496.65 |
| `results/duet_tree_al_full` | 83 / 35.57 | 0 / 0 |
| `results/duet_tree_posthoc` | 75 / 22.94 | 1,199 / 1,418.05 |
| `results/duet_tree_followup` | 78 / 4.88 | 48 / 23.22 |
| `ssd/experiments/proxy_source_ablation` | 177 / 7.93 | 435 / 19,273.28 |

예를 들어 posthoc `tree_metrics.jsonl`은 약136MB로 GitHub 단일 파일 제한보다
크다. 이 파일과 `subset_metrics.jsonl`, `prospective_metrics.jsonl`, 두 run의
`trees.jsonl` 및 raw NPZ는 archive에 있다. `.gitignore`와 파일 manifest가 이를
구분하므로 `git clone`만으로 전체 raw가 복원됐다고 생각하지 않는다.

생성한 archive의 현재 서버 경로:

```text
/home/chokwans99/PSD/handoff_artifacts/duet_research_raw_20260927.tar
```

`handoff_artifacts/` 자체는 Git에서 제외된다. Archive를 push한 것이 아니다.
새 서버 hostname/경로가 아직 없으므로 원격 전송은 수행하지 않았다.

## 전송과 복원

Git clone 뒤 새 서버의 저장소 루트에서 실행한다. 아래 hostname은 실제 이전
서버 주소로 바꿔야 한다. 상대 경로를 보존하므로 기존 절대경로를 포함한 metadata는
그대로 역사적 기록이며, 새 실험의 model/data 경로는 새 config에서 지정한다.

```bash
mkdir -p handoff_artifacts
rsync -avP OLD_SERVER:/home/chokwans99/PSD/handoff_artifacts/duet_research_raw_20260927.tar handoff_artifacts/
(
  cd handoff_artifacts
  sha256sum -c ../results/handover_20260927/archive_SHA256SUMS
)
tar --keep-old-files -xf handoff_artifacts/duet_research_raw_20260927.tar -C .
python3 results/handover_20260927/manage_artifacts.py verify --group all
```

새 clone의 원시 파일이 없는 상태에서 복원한다. `--keep-old-files`는 이미 존재하는
파일을 덮어쓰지 않도록 한다. 중간에 실패한 복원을 재시도한다면 먼저 존재 파일의
checksum을 확인하고 필요한 누락분만 복구한다. 전체 archive와 풀린 raw를 동시에
보관하려면 약47GB 이상의 추가 공간에 여유를 더해 확보한다.

Archive를 만들지 않고 개별 파일을 전송할 경우 manifest의 정확한 목록으로 복사한다.
예를 들어 새 서버의 repo root에서:

```bash
rsync -avP --files-from=results/handover_20260927/archive_files.txt \
  OLD_SERVER:/home/chokwans99/PSD/ ./
python3 results/handover_20260927/manage_artifacts.py verify --group all
```

구체적 경로와 checksum 파일의 역할은 다음과 같다.

- `archive_files.txt`: tar에 들어간 원본 상대경로 목록.
- [`raw_SHA256SUMS`](raw_SHA256SUMS): 각 원본 파일의 SHA256 목록.
- [`archive_SHA256SUMS`](archive_SHA256SUMS): tar 파일 한 개의 SHA256.
- `artifact_manifest.json`: group/bytes/hash/reason을 포함한 전체 inventory.

## Git/archive에 포함하지 않은 외부 의존성

| 의존성 | 기존 위치/상태 | 새 서버 작업 |
|---|---|---|
| Dense LayerSkip Llama2-70B | `/data/chokwans99/models/layerskip-llama2-70B`를 env가 참조 | 실제 원본 checkpoint 준비 및 hash/dtype 확인 |
| 기존 AWQ target | `/home/chokwans99/awq_calibrated/layerskip_llama2_70b` | 역사적 AWQ 재현이 필요할 때만 별도 전송 |
| AWQ TP4 artifact | `/home/chokwans99/awq_artifacts/layerskip70b_awq_tp4` | Dense에는 사용하지 않음; 다른 TP면 artifact 호환성 확인 |
| Draft | `/data/chokwans99/models/TinyLlama-1.1B-Chat-v1.0` | Tokenizer/config/weights 함께 준비 |
| 초기 소규모 datasets | `/home/chokwans99/ssd_datasets/processed_datasets`, `/data/ssd_datasets` | 초기 실험 재실행 시 필요; full480 질문은 repo에 포함 |
| Paper benchmark runner | `<HOME>/baseline/runners/`, `DUET_PAPER_RESULTS/scripts` | 논문 protocol 비교에 쓸 때 별도 준비 |
| Python 환경·CUDA 확장 cache | `ssd/.venv`, HF/JIT/kernel cache | 복사 대신 새 장비에서 재구성 |

모델 weight 파일은 이 archive에 포함하지 않는다. 새 서버에서 offline model load를
할 때 tokenizer/config/safetensors shard/index가 모두 있는지 확인한다. 동일 모델
이름이어도 weights가 다르면 별도 실험이다. 외부 경로의 size/mtime는 원래 queue
manifest에 남아 있지만 새 서버 모델 무결성 증명으로 대신 쓰지 않는다.

`__pycache__`, `.pyc/.pyo`, queue lock/temp 등 재생성 가능한 파일은 양쪽에서 제외했다.
제외 목록은 manifest의 `omitted_regenerable`에 있다. Run의 JSON PID/status/log는
실험 이력으로 포함하며, clone했다고 이전 프로세스가 재시작되는 것은 아니다.

## Inventory 재생성 시 주의

`manage_artifacts.py inventory`는 현재 파일 기준으로 새 manifest를 작성한다.
수령한 frozen manifest를 바꿔 mismatch를 없애는 용도로 실행하지 않는다.
수령 검증에는 `verify`만 쓴다. 새 campaign의 inventory는 새 디렉터리에 별도로
만든다. 역사적 `final_manifest.json`과 `execution_manifest.json`도 그대로 보존한다.
