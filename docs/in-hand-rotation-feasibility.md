# In-Hand Rotation の移植検討

[Msornerrrr/in-hand-rotation-mjlab](https://github.com/Msornerrrr/in-hand-rotation-mjlab)
（LEAP ハンドによる手内キューブ回転）を mjswan_playground のタスクとして追加できるか。

- 調査日: 2026-08-21
- 対象: `14004fe` (2026-02-20)
- 検証環境: Python 3.12 / mjlab 1.5.3 / mjswan main / onnxruntime 1.29.0

## 結論

**可能。ただしドロップインではない。** 障壁は上流リポジトリではなく mjswan の行動層にある。

観測・シーン・方策・終了条件は既存の mjswan でそのまま通る。通らないのは行動項がひとつだけ状態を持つ点
── この方策は関節位置の目標値を毎ステップ積分する差分制御で、mjswan の行動層は設計上すべて無状態
（TS ネイティブ実装、ADR 0005 §7）。wbc-g1 のために `ReferenceJointPositionActionCfg` を足したのと
同じ規模の作業が 1 件必要になる。

| | |
|---|---|
| Actor 観測項のうちスロットが既にあるもの | 2 / 2 のうち 1（もう 1 つは要拡張） |
| mjswan 本体に必要な拡張 | 2 件 |
| playground 側で書く作業 | 6 件 |
| 想定工数 | 3–5 日 |

## 載せる対象

LEAP ハンド（16 自由度・固定基部）が手のひらでキューブを回し続ける。HORA の報酬設計を mjlab 上に
再実装したもので、非対称 Actor-Critic とドメインランダム化で実機転移まで到達している。
学習済みチェックポイントが同梱されているのは `Mjlab-Leap-Left-Custom-HandCube-Rotate` のみ。

ブラウザに必要なのは Actor 側だけ。Critic の 11 項（キューブ姿勢・質量・摩擦などの特権情報）は
学習専用で実機にも出て行かない。**Actor が読むのは 2 項だけ**である。

```
       t−9  t−8  t−7  t−6  t−5  t−4  t−3  t−2  t−1   t
joint_pos   ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓    ✓   (16)  ← 既存スロットで配れる
cmd_pos     !    !    !    !    !    !    !    !    !    !   (16)  ← 要ネイティブ観測

10 × 32 = 320 → MLP 512·512·256 (ELU) → 16 actions ∈ [−1,1]
             → q ← clip(q + δ)  ※要実装 → ideal-PD → MuJoCo
```

チェックポイントを実際に読み込み、入力 320・出力 16 であることを確認済み。

## レイヤ別の可否

### そのまま動く（追加実装なし）

- **シーン** — ハンド + キューブ + 平面。`slotReader` のエンティティ索引は接頭辞で汎用に解決するので
  2 体目の `cube` も普通に読める
- **物理設定** — `timestep 0.005` / `elliptic` cone / `impratio 10` は spec に載せて運べる
- **アクチュエータ** — ideal-PD。mjswan の `resolve_pd_gains` がそのまま効く
- **Actor 観測 ①** — `joint_pos_rel(biased=True)` は mjlab の関数をそのままトレースでき、
  `joint_pos_biased` スロットも実装済み
- **10 ステップ履歴** — 群レベルの `history_length=10` を項ごとの `history_steps=(9…0)` に展開する
  （husky-skater と同じ手）
- **終了条件** — キューブ落下（高さ）・速度超過はどちらも既存スロットで書ける
- **制御周期** — 20 Hz（`0.005 × 10`）

### playground 側で書く

- **ONNX 化** — 同梱の `.pt` から Actor + 正規化層を書き出す。**検証済**: 2.25 MB / 320→16、
  onnxruntime と PyTorch の差は最大 1.5e-4
- **初期姿勢のキーフレーム化** — grasp cache（7,700 サンプル × 把持関節角 16 + キューブ相対姿勢 7）から 1 つ選ぶ
- **行動クリップ** — 学習側の `clip_actions=1.0` を明示的に持ち込む
- **不要な層の切り落とし** — Critic 観測 11 項・報酬・カリキュラム・メトリクス・可視化専用コマンド・接触センサ
- **アセット同梱** — OBJ メッシュ 17 MB（`.mjz` 圧縮後はこれより大幅に小さく、Cloud の 50 MB/ファイル制限内）
- **README と忠実度チェック** — `run_parity` で項ごとに突き合わせ

### mjswan 本体の拡張が必要 ← 唯一の実質的な障壁

**BLOCKER 01 — 差分関節位置の行動項**

出力は目標角ではなく 1 ステップあたり ±1/24 rad の増分。前ステップの目標値に積分し、ソフト関節限界で
クランプして初めて指令になる。リセット時は現在の関節角から初期化。

現状 `applyAction.ts` の `ControlType` は `joint_position` / `joint_position_reference` / `torque` /
`muscle_activation` の 4 種で、いずれも `base + scale · a` の無記憶な式。

必要な作業: Python 側の cfg クラス、TS 側の状態つき項（目標値保持・リセット・ソフト限界クランプ）、
ソフト関節限界のバンドル、任意で decimation 補間。

**BLOCKER 02 — 指令位置の観測項**

Actor の 2 項目 `joint_pos_commanded` は行動マネージャの内部目標値そのもの。エンティティ状態でも
センサでもないので、トレーサのスロット（entity data / sensor / command state）のどれにも当てはまらない。

解き方: BLOCKER 01 の行動項がブラウザ側で目標値を保持するので、それを `last_action` と同じ
ネイティブ観測マーカーとして出す。10 ステップ履歴は `HistoryObservation` が graph 由来と
ネイティブ由来の両方を扱えるため追加対応は不要。

ONNX に畳み込む回避策は成立しない（積分値はクランプ後の値であり、生の行動の総和ではないため）。

### 移植しない

- **ドメインランダム化一式** — 観測ノイズ・遅延・質量・摩擦・PD ゲイン・アクチュエータ遅延。
  mjswan は学習専用フィールドとして受理・無視する（husky-skater も同じ扱い）
- **grasp cache からの再サンプリング** — リセットごとにキューブ寸法を変える処理はブラウザ側に
  対応物がない。寸法は固定になる
- **姿勢逸脱の終了条件** — リセット時姿勢を保持する状態つきクラスでトレース不可。落下判定で足りる

## 上流リポジトリは mjlab に追従していない

最終コミットは 2026-02-20、当時の mjlab は v1.1 系。以来 mjlab は DR API を `mdp.dr.*` 名前空間へ
再編しており、**現在の mjlab では import が通らない**。mjlab 1.5.3 を入れて実際に確認した:

```
$ pip install mjlab==1.5.3 && python -c "import in_hand_rotation_mjlab.tasks"
  File ".../robots/leap_hand/leap_right_constants.py", line 19, in <module>
    from mjlab.actuator import DelayedActuatorCfg, IdealPdActuatorCfg
ImportError: cannot import name 'DelayedActuatorCfg' from 'mjlab.actuator'
```

AST で全 mjlab 参照を走査した結果、**61 シンボルが解決、6 シンボルが不整合**:

| 参照（v1.1 系） | 状態 | mjlab 1.5.3 での対応物 | 影響範囲 |
|---|---|---|---|
| `mjlab.actuator.DelayedActuatorCfg` | 消滅 | `ActuatorCfg` の `delay_*` フィールド群 | ロボット定数（import 失敗点） |
| `mjlab.terrains.TerrainImporterCfg` | 改名 | `TerrainEntityCfg` | シーン定義 |
| `mjlab.utils.os.update_assets` | 消滅 | `mjlab.utils.spec` の meshdir 処理 | メッシュ埋め込み |
| `envs_mdp.randomize_field` | 再編 | `mdp.dr.body_ipos` / `dof_damping` ほか | DR イベント 4 件 |
| `envs_mdp.randomize_pd_gains` | 再編 | `mdp.dr.pd_gains` | DR イベント 1 件 |
| `envs_mdp.sync_actuator_delays` | 消滅 | アクチュエータ cfg に吸収（イベント不要） | DR イベント 1 件 |

6 件のうち 4 件は DR イベント — どのみち移植しない部分である。つまりこの不整合は、
**上流パッケージを import する統合方式を選んだ場合にだけ問題になる**。

## 統合方式

### Option A — wbc-g1 方式（パッケージを入れる）

optional extra で in-hand-rotation-mjlab を入れ、登録された mjlab タスクを `add_scene_mjlab()` に渡す。
env cfg から全項が自動で埋まる。

- 利点: タスク定義が上流と一本化される。設定のコピーが発生しない
- 条件: 上流の mjlab 追従（6 シンボル）が前提。PyPI 未公開なので git 依存になる。追従は他人のリポジトリ側の作業

### Option B — husky-skater 方式（データだけ借りる）★推奨

コミット固定で `.cache/` に clone し、パッケージは一切 import しない。使うのはハンドの XML とメッシュ、
grasp cache、チェックポイントの 3 つだけ。シーンと MDP は playground 側で組み立てる。

- 利点: mjlab 追従問題を丸ごと回避できる。Actor 観測が 2 項しかないので手組みのコストが小さい
- 条件: シーン spec（ハンド + キューブ + 平面 + キーフレーム + ソルバ設定）を自前で組む

**B を推す理由**は 2 つ。第一に、A は他人のリポジトリが mjlab に追いつくのを待つことになる
（6 シンボルの修正 PR を出す手はあるが、マージのタイミングは制御できない）。第二に、このタスクは
Actor 観測が 2 項しかないため、手組みの負担が husky-skater より軽い。

なお **A・B いずれを選んでも BLOCKER 01・02 は残る**（mjswan の行動層は設計上どちらの経路でも共通）。

## 実装計画

依存関係が実際に順序を持つ。1 が終わるまで 3 の動作確認はできない。

### 1. mjswan に差分行動項を足す（1–2 日、ttktjmt/mjswan）

- `JointPositionDeltaActionCfg`（Python 側 cfg・シリアライズ）
- `applyAction.ts` に状態つき項 — 目標値の保持、リセット時の現在角からの初期化、ソフト関節限界クランプ
- ソフト関節限界を policy config に載せる（ブラウザは自前でモデルをコンパイルするため）
- 指令位置のネイティブ観測マーカー（`last_action` の実装に倣う）
- 任意: decimation 内の目標値線形補間（`interpolate_decimation` 相当）
- 単体テストと `rolloutParity` フィクスチャ

### 2. アセットを用意する（0.5 日、手順は検証済み）

- Actor + 正規化層 → ONNX（実行確認済み、20 行程度）
- grasp cache から代表把持を 1 つ選び、キーフレームとして焼く
- ハンド XML とメッシュのパス書き換え・キューブと平面の spec 合成

### 3. playground タスクとして組む（1–2 日、ttktjmt/mjswan_playground）

- `src/mjswan_playground/leap_inhand/` — `main.py` / `terms.py` / `README.md`
- `registry.py` と `_deps.py` にコミット固定の clone を登録
- 忠実度チェック — 上流の env と項ごとに突き合わせ、README に `max |Δ|` を記載（husky-skater の書式）
- ブラウザでの実時間性能を実測

## リスク

**［高］ロールアウトの発散** — 接触が支配的なタスクは本質的にカオス的で、項ごとのパリティが通っても
軌跡は数秒で分かれる。歩行タスクとは事情が違い、「上流と同じ絵が出る」ことは保証できない。
救いは、この方策が重いドメインランダム化の下で学習され実機転移まで届いていること。
README にはパリティの意味（項ごとの一致であって軌跡の一致ではない）を明記すべき。

**［中］ブラウザでの実時間性** — 物理 200 Hz、elliptic cone、`impratio 10`、ソルバ 10 反復。
手と立方体だけなので規模は小さいが、接触点数が多く solver は重い側。既存 2 タスクは humanoid で
条件が違うため、Phase 3 で実測が要る。

**［中］操作対象がない** — このタスクにはユーザコマンドが一切ない（ただ回し続ける方策）。
既存 2 タスクのようなスライダーが作れないため、インタラクションはキューブをドラッグして外乱を与える・
リセットするに限られる。もっとも手内操作のデモとしてはむしろ分かりやすい。
grasp cache が 5 段階のキューブ寸法を持つので、寸法違いを複数シーンとして並べる案もある。

**［低］ライセンスと礼儀** — リポジトリ本体は Apache-2.0、LEAP ハンドのモデルは MIT（LEAP Hand Sim 由来）、
タスク設計は HORA に基づく。いずれも利用は許諾されている。ただし husky-skater で著者の許諾を明記した
前例に倣い、公開前に著者へ一報を入れるのが望ましい。README には in-hand-rotation-mjlab /
LEAP Hand Sim / HORA の 3 者を明記する。

## この結論の根拠（すべて実行済み）

1. **3 リポジトリの取得** — in-hand-rotation-mjlab（`14004fe`）、mjswan（main）、mjlab（全タグ）
2. **mjlab 1.5.3 環境での import 再現** — venv に `mjlab==1.5.3` を導入し
   `in_hand_rotation_mjlab.tasks` の import を実行 → `DelayedActuatorCfg` の ImportError を再現
3. **API 整合の全走査** — 全 `.py` を AST で解析し `from mjlab.… import X` と `envs_mdp.X` を
   導入済み mjlab に照合（61 解決 / 6 不整合）。mjlab の全タグを横断して DR API が v1.2.0 で
   `mdp.dr.*` へ再編されたことを特定
4. **チェックポイントの実測** — Actor が `320 → 512 → 512 → 256 → 16`、正規化層が 320 次元であることを確認。
   観測レイアウトの推定（10 × (16+16) = 320）と一致
5. **ONNX 化の実演** — Actor と正規化層を書き出し（2,253,623 バイト）、onnxruntime で推論して
   PyTorch と照合 → 最大差 1.5e-4（float32）
6. **mjswan 側のソース精読** — `applyAction.ts`（行動 4 種）、`slotReader.ts`（配れる entity data
   フィールドとエンティティ索引の汎用性）、`compile/tracer.py`（スロットの 3 名前空間）、
   `_onnx_build.py`（ネイティブ観測の扱い）、ADR 0005、CONTEXT.md

### 実測値の一覧

| 項目 | 値 | 出所 |
|---|---|---|
| Actor 観測次元 | 320 | チェックポイント実測 |
| 行動次元 | 16 | チェックポイント実測 |
| 制御周期 | 20 Hz (0.005 × 10) | env cfg |
| 1 ステップの関節増分 | ±1/24 rad | 行動項 cfg |
| ONNX サイズ | 2.25 MB | 書き出し実測 |
| ONNX ↔ PyTorch 最大差 | 1.5e-4 | onnxruntime 実行 |
| メッシュ（OBJ 16 点） | 17 MB | リポジトリ実測 |
| grasp cache | 7,700 サンプル | npz ヘッダ |
| mjlab 参照 解決 / 不整合 | 61 / 6 | AST 走査 |
| 上流の最終コミット | 2026-02-20 | git log |
