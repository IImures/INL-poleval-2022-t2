# Project Plan: Polish Abbreviation Disambiguation CLI

## 1. Project Goal

Build a simple CLI application for the **PolEval 2022 abbreviation disambiguation task**.

The task is to decide whether a given phrase is an abbreviation and, if yes, return:

1. its expanded inflected form,
2. its base form.

### Example

**Input:**

```text
abbr: pkt. proc.
context: proc. Kolejne 0,12 pkt. proc. wynika ..., a 0,08 <mask> z zaburzeń ...
```

**Expected output:**

```text
punktu procentowego    punkt procentowy
```

The official task data contains:

- abbreviation,
- expanded form,
- base form,
- context with a `<mask>` placeholder.

The test data contains only:

- abbreviation,
- context.

The system must predict:

- expanded inflected form,
- base form.

### Official Evaluation

The official evaluation uses two accuracy measures:

```text
Af = accuracy of expanded inflected forms
Ab = accuracy of base forms

Final score = 0.25 * Af + 0.75 * Ab
```

Because of this, predicting the **base form** is more important than predicting the inflected form.

---

## 2. Project Type

The project will be a simple **Python CLI application**.

No complicated deployment is needed.

### Recommended Project Structure

```text
abbreviation-disambiguation/
│
├── data/
│   ├── train/
│   ├── dev-0/
│   ├── test-A/
│   └── test-B/
│
├── models/
│   ├── base_model.joblib
│   ├── expanded_model.joblib
│   ├── majority_dictionary.json
│   └── metadata.json
│
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md
```

The project can be split into two main Python files.

### `train_model.py`

Used for:

- loading training data,
- preprocessing text,
- training the model,
- evaluating on validation data,
- saving the trained model.

### `predict.py`

Used for:

- loading the trained model,
- reading test/input data,
- predicting expanded and base forms,
- saving predictions to `.tsv`.

---

## 3. Recommended Technical Approach

Use a practical machine learning approach instead of a heavy neural model.

Strong competition systems used encoder-decoder models like **ByT5** and **plT5**, where the model receives context with the abbreviation and generates both the base and inflected forms.

However, for a simple CLI project, a classical ML model is easier to implement, faster to train, and easier to explain.

### Recommended Approach

```text
TF-IDF features + Logistic Regression / Linear SVM
```

Train two models:

| Model | Purpose |
|---|---|
| `model_expanded_form` | Predicts the inflected expanded form |
| `model_base_form` | Predicts the base form |

Input features should include:

```text
abbreviation + context
```

Example internal input:

```text
ABBR=pkt. proc. CONTEXT=proc. Kolejne 0,12 pkt. proc. wynika ..., a 0,08 <mask> z zaburzeń ...
```

This lets the model learn both:

- what abbreviation is being expanded,
- how surrounding words affect the correct meaning and grammatical form.

---

## 4. Libraries Used

## 4.1 `pandas`

Used for reading and writing `.tsv` or `.csv` files.

### Why it is needed

- PolEval data is tabular.
- Training data contains columns like abbreviation, expanded form, base form, and context.
- Test data contains abbreviation and context.

### Example

```python
import pandas as pd

df = pd.read_csv("data/train/train.tsv", sep="\t")
```

---

## 4.2 `scikit-learn`

Main machine learning library.

Used for:

- TF-IDF vectorization,
- train/validation split,
- Logistic Regression or Linear SVM,
- accuracy calculation,
- building pipelines.

Recommended components:

```python
TfidfVectorizer
LogisticRegression
LinearSVC
Pipeline
FeatureUnion
train_test_split
accuracy_score
```

### Why it is good here

- easy to use,
- fast on small/medium datasets,
- no GPU required,
- simple to explain in documentation.

---

## 4.3 `joblib`

Used for saving and loading trained models.

### Example

```python
import joblib

joblib.dump(model, "models/base_model.joblib")
model = joblib.load("models/base_model.joblib")
```

### Why it is needed

After training, the CLI should be able to reuse the model without retraining.

---

## 4.4 `argparse`

Used to create a CLI interface.

### Example Commands

```bash
python train_model.py --train data/train/train.tsv --dev data/dev-0/dev.tsv --model-dir models/
```

```bash
python predict.py --input data/test-A/in.tsv --model-dir models/ --output predictions.tsv
```

### Why it is needed

- simple command-line usage,
- no web server,
- no Docker required.

---

## 4.5 `re`

Used for simple text preprocessing with regular expressions.

Possible uses:

- normalizing spaces,
- checking whether abbreviation ends with `.`,
- cleaning duplicated whitespace,
- replacing `<mask>` consistently.

### Example

```python
import re

text = re.sub(r"\s+", " ", text).strip()
```

---

## 4.6 `json`

Used for saving metadata.

### Example metadata

```json
{
  "model_type": "tfidf_logistic_regression",
  "train_file": "data/train/train.tsv",
  "features": ["abbreviation", "context"],
  "score_formula": "0.25 * Af + 0.75 * Ab"
}
```

### Why it is useful

- documents how the model was trained,
- makes results reproducible.

---

## 4.7 Optional: `transformers`

Use only as an advanced version.

Possible models:

- `allegro/plt5-base`,
- `google/byt5-small`,
- `google/mt5-small`.

Top PolEval solutions used T5-style encoder-decoder models, including ByT5 and plT5.

However, for this CLI project, this should be optional because:

- training is slower,
- setup is heavier,
- GPU may be needed,
- deployment is less simple.

---

## 4.8 Optional: `morfeusz2`

Morfeusz is a Polish morphological analyzer.

Possible use:

- improve dictionary-based fallback,
- analyze Polish inflection,
- gather additional abbreviation candidates.

For the first version, it is optional.

---

## 5. Data That Should Be Gathered

## 5.1 Official PolEval Data

Required data:

```text
train/
dev-0/
test-A/
test-B/
```

Training data should contain:

- abbreviation,
- expanded inflected form,
- base form,
- context with `<mask>`.

Test data should contain:

- abbreviation,
- context with `<mask>`.

---

## 5.2 Additional Dictionary Data

Useful additional sources:

- Polish abbreviation dictionaries,
- Wiktionary abbreviation entries,
- SJP abbreviation entries,
- Morfeusz dictionary entries.

This data can be used to build a fallback dictionary:

```text
abbr. -> possible expanded forms
```

### Example

```text
wyd. -> wydanie / wydawca / wydawnictwo / wydawniczy
woj. -> województwo / wojewoda / wojewódzki / wojenny / wojskowy
```

---

## 5.3 Extra Context Data

Optional data:

- Polish press articles,
- Polish Wikipedia,
- news corpora.

### Why it can help

- abbreviation meaning depends heavily on context,
- more context examples can improve model generalization,
- synthetic examples can be generated by replacing full forms with abbreviations.

---

## 5.4 Data to Store Locally

Recommended local files:

```text
data/raw/train.tsv
data/raw/dev.tsv
data/raw/test.tsv
data/processed/train_processed.tsv
data/processed/dev_processed.tsv
data/dictionaries/abbr_dictionary.tsv
```

Example dictionary format:

```text
abbreviation    expanded_form             base_form           source
pkt.            punktu                    punkt               train
pkt. proc.      punktu procentowego       punkt procentowy    train
woj.            województwo               województwo         sjp
```

---

## 6. Input and Output Format

## 6.1 Training Input

Example training row:

```text
abbr    expanded    base      context
s.      sobota      sobota    Karpaty Siepraw ... Grybovia (<mask> 16) ...
```

The context contains `<mask>`, which marks the original abbreviation position.

---

## 6.2 Prediction Input

Example test row:

```text
abbr    context
s.      Karpaty Siepraw ... Grybovia (<mask> 16) ...
```

---

## 6.3 Prediction Output

Output should contain:

```text
expanded_form    base_form
```

Example:

```text
sobota                  sobota
punktu procentowego     punkt procentowy
```

---

## 7. Model Design

## 7.1 Baseline 1: Majority Dictionary

Before machine learning, implement a simple baseline.

For each abbreviation from training data:

```text
choose the most common expanded form
choose the most common base form
```

### Example

```text
s. -> sobota / sobota
n. -> niedziela / niedziela
```

### Advantages

- very easy,
- strong baseline for frequent abbreviations,
- useful fallback when ML confidence is low.

### Disadvantages

- does not understand context,
- weak for ambiguous abbreviations like `p.`, `w.`, `s.`, `m.`.

---

## 7.2 Baseline 2: TF-IDF + Classifier

Train two classifiers:

```text
classifier_expanded
classifier_base
```

Input text:

```text
ABBR=<abbr> CONTEXT=<context>
```

Output labels:

```text
expanded form
base form
```

Recommended feature extraction:

```python
TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    min_df=1
)
```

### Why character n-grams

- useful for Polish inflection,
- can handle rare words better,
- works well with abbreviations and short forms.

Alternative:

```python
TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2)
)
```

Best simple option:

```text
combine word n-grams + character n-grams
```

This can be done with `FeatureUnion`.

---

## 7.3 Recommended Final Model

Use:

```text
FeatureUnion(
    word-level TF-IDF,
    char-level TF-IDF
)
+
LogisticRegression
```

Train separately:

```text
expanded_model
base_model
```

Final prediction:

```python
expanded = expanded_model.predict(input_text)
base = base_model.predict(input_text)
```

If the model predicts something impossible or empty, use the majority dictionary fallback.

---

## 8. Training Script Plan: `train_model.py`

## 8.1 CLI Arguments

The script should support:

```bash
python train_model.py \
  --train data/train/train.tsv \
  --dev data/dev-0/dev.tsv \
  --model-dir models/
```

Arguments:

| Argument | Description |
|---|---|
| `--train` | Path to training file |
| `--dev` | Optional validation file |
| `--model-dir` | Output directory for trained models |

---

## 8.2 Steps Inside `train_model.py`

### Step 1: Load data

Use pandas:

```python
train_df = pd.read_csv(train_path, sep="\t")
```

Expected columns:

- `abbr`,
- `expanded`,
- `base`,
- `context`.

If official column names are different, rename them internally.

---

### Step 2: Validate data

Check:

- no missing abbreviation,
- no missing context,
- no missing expanded form,
- no missing base form,
- context contains `<mask>`.

If something is missing, print a warning and skip the row.

---

### Step 3: Preprocess text

Create one input field:

```python
df["input_text"] = "ABBR=" + df["abbr"] + " CONTEXT=" + df["context"]
```

Normalize:

- lowercase text,
- strip spaces,
- replace multiple spaces with one space,
- keep Polish characters,
- keep dots in abbreviations,
- keep `<mask>`.

Do not remove punctuation completely because dots are important for abbreviations.

---

### Step 4: Build majority dictionary

Create:

```python
majority_expanded[abbr] = most_common_expanded_form
majority_base[abbr] = most_common_base_form
```

Save it as:

```text
models/majority_dictionary.json
```

---

### Step 5: Train expanded-form model

Input:

```text
X = input_text
y = expanded
```

Train pipeline:

```python
Pipeline([
    ("features", FeatureUnion([
        ("word_tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2))),
        ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)))
    ])),
    ("clf", LogisticRegression(max_iter=1000))
])
```

---

### Step 6: Train base-form model

Input:

```text
X = input_text
y = base
```

Use the same pipeline structure.

---

### Step 7: Evaluate on dev data

Predict:

```python
pred_expanded = expanded_model.predict(X_dev)
pred_base = base_model.predict(X_dev)
```

Calculate:

```python
Af = accuracy_score(y_expanded_dev, pred_expanded)
Ab = accuracy_score(y_base_dev, pred_base)
final_score = 0.25 * Af + 0.75 * Ab
```

Print:

```text
Expanded accuracy Af: ...
Base accuracy Ab: ...
Final score: ...
```

---

### Step 8: Save models

Save:

```text
models/expanded_model.joblib
models/base_model.joblib
models/majority_dictionary.json
models/metadata.json
```

---

## 9. Prediction Script Plan: `predict.py`

## 9.1 CLI Arguments

The script should support:

```bash
python predict.py \
  --input data/test-A/in.tsv \
  --model-dir models/ \
  --output predictions.tsv
```

Arguments:

| Argument | Description |
|---|---|
| `--input` | Path to test/input file |
| `--model-dir` | Directory with trained models |
| `--output` | Output file path |

---

## 9.2 Steps Inside `predict.py`

### Step 1: Load models

Load:

```python
expanded_model = joblib.load("models/expanded_model.joblib")
base_model = joblib.load("models/base_model.joblib")
```

Load dictionary fallback:

```python
majority_dictionary = json.load(...)
```

---

### Step 2: Load input data

Expected columns:

- `abbr`,
- `context`.

---

### Step 3: Preprocess input

Use the same preprocessing as in training.

Important:

> Training and prediction preprocessing must be identical.

---

### Step 4: Predict forms

For each row:

```python
expanded = expanded_model.predict([input_text])[0]
base = base_model.predict([input_text])[0]
```

---

### Step 5: Apply fallback

If prediction is empty or invalid:

```text
use majority dictionary for this abbreviation
```

If abbreviation was not seen during training:

```text
return abbreviation itself as both expanded and base form
```

This is useful because the task also includes non-abbreviation cases where the correct output can be the original phrase itself.

---

### Step 6: Save output

Save as `.tsv`:

```text
expanded_form    base_form
```

---

## 10. Step-by-Step Project Implementation Plan

## Step 1: Create project repository

Create folder:

```bash
mkdir abbreviation-disambiguation
cd abbreviation-disambiguation
```

Create files:

```bash
touch train_model.py predict.py requirements.txt README.md
mkdir data models
```

---

## Step 2: Download official PolEval data

Download or clone the official repository.

Required folders:

```text
train
dev-0
test-A
test-B
```

---

## Step 3: Inspect the data format

Open training files and check:

- separator: comma or tab,
- column names,
- encoding,
- number of rows,
- missing values.

Write down the final column mapping:

```text
official abbreviation column -> abbr
official expanded column     -> expanded
official base column         -> base
official context column      -> context
```

---

## Step 4: Implement data loading

In `train_model.py`, create:

```python
def load_training_data(path):
    ...
```

It should:

- read file,
- rename columns,
- remove invalid rows,
- return dataframe.

---

## Step 5: Implement preprocessing

Create function:

```python
def preprocess_row(abbr, context):
    ...
```

It should return:

```text
ABBR=<abbr> CONTEXT=<context>
```

Rules:

- strip spaces,
- normalize multiple spaces,
- lowercase,
- keep `<mask>`,
- keep dots.

---

## Step 6: Implement majority baseline

Create:

```python
def build_majority_dictionary(df):
    ...
```

It should produce:

```json
{
  "s.": {
    "expanded": "sobota",
    "base": "sobota"
  }
}
```

Evaluate this baseline first.

This is important because it gives a simple reference result before ML.

---

## Step 7: Implement official scoring function

Create:

```python
def calculate_score(true_expanded, pred_expanded, true_base, pred_base):
    ...
```

Use:

```text
Af = expanded accuracy
Ab = base accuracy
score = 0.25 * Af + 0.75 * Ab
```

---

## Step 8: Train first ML model

Train only the base-form model first.

Reason:

```text
base form has higher weight in final score
```

Use:

```text
TF-IDF + Logistic Regression
```

Save the result.

---

## Step 9: Train expanded-form model

Use the same approach for expanded forms.

This model may be harder because Polish inflection depends on grammar and context.

---

## Step 10: Evaluate on dev data

Compare:

- majority baseline,
- ML base model,
- ML expanded model,
- combined score.

Create a simple results table:

| Model | Af | Ab | Final |
|---|---:|---:|---:|
| Majority baseline | ... | ... | ... |
| TF-IDF + Logistic Regression | ... | ... | ... |

---

## Step 11: Analyze errors

Check wrong predictions manually.

Focus on:

- ambiguous abbreviations,
- rare abbreviations,
- non-abbreviations,
- multiword abbreviations,
- wrong grammatical case.

---

## Step 12: Improve features

Try adding:

- abbreviation repeated separately,
- left context,
- right context,
- words near `<mask>`,
- context window of 5-10 words around `<mask>`.

Example input:

```text
ABBR=s. LEFT=Grybovia RIGHT=16 Orkan CONTEXT=...
```

This may help because nearby words are often more important than the whole context.

---

## Step 13: Add dictionary fallback

If abbreviation exists in majority dictionary and model confidence is low, use dictionary answer.

For Logistic Regression, confidence can be estimated with:

```python
predict_proba()
```

Example rule:

```python
if confidence < 0.40:
    use_majority_dictionary()
```

---

## Step 14: Implement `predict.py`

The script should:

- load model,
- load input file,
- preprocess rows,
- predict expanded and base forms,
- save output file.

Example usage:

```bash
python predict.py --input data/test-A/in.tsv --model-dir models --output predictions.tsv
```

---

## Step 15: Prepare README

README should contain:

- project description,
- task description,
- data format,
- installation,
- training command,
- prediction command,
- evaluation command,
- known limitations.

---

## 11. Possible Advanced Improvements

## 11.1 Transformer model

Use a sequence-to-sequence model:

```text
input:  "abbr: s. context: ... <mask> ..."
output: "sobota <sep> sobota"
```

Possible models:

- `plT5`,
- `ByT5`,
- `mT5`.

This follows the approach used by strong PolEval systems, where the encoder receives the context and the decoder generates the base and inflected forms.

However, this should be treated as an advanced version.

---

## 11.2 Synthetic data generation

Create extra examples:

1. collect full Polish phrases,
2. replace them with abbreviations,
3. train model to recover original forms.

Example:

```text
punkt procentowy -> pkt. proc.
godzina -> godz.
województwo -> woj.
```

---

## 11.3 Ensemble

Train several models:

- Logistic Regression,
- Linear SVM,
- Naive Bayes.

Then use majority voting.

---

## 12. Minimal `requirements.txt`

```text
pandas
scikit-learn
joblib
```

Optional:

```text
transformers
torch
datasets
morfeusz2
```

---

## 13. Final Recommended Scope

For a clean and realistic CLI project, implement this version:

1. Official PolEval data loader,
2. Majority dictionary baseline,
3. TF-IDF + Logistic Regression model,
4. Separate models for expanded form and base form,
5. Official scoring function,
6. CLI training script,
7. CLI prediction script,
8. Error analysis section in README.

