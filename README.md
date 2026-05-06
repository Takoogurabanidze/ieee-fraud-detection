# IEEE-CIS Fraud Detection

## კონკურსის მიმოხილვა

IEEE-CIS Fraud Detection არის Kaggle-ის კომპეტიცია, სადაც უნდა განისაზღვროს, თაღლითურია თუ არა კონკრეტული ტრანზაქცია. დათასეთი მოიცავს 590,000-ზე მეტ ტრანზაქციას 400-ზე მეტი feature-ით. შეფასება ხდება ROC-AUC მეტრიკით.

მონაცემთა ბაზა ორი ფაილისგან შედგება: transaction და identity, რომლებიც TransactionID-ით ერთმანეთთან არის დაკავშირებული. მთავარი სირთულეები იყო  დისბალანსი (fraud-ი მხოლოდ 3.5%-ია), ბევრი გამოტოვებული მნიშვნელობა და 339 V სვეტი, რომლებიც ძლიერ კორელაციაში არიან ერთმანეთთან.

```
Train:  590,540 ტრანზაქცია | 431 სვეტი
Test:   506,691 ტრანზაქცია
Fraud:  ~3.5%
Metric: ROC-AUC
```

## მიდგომა

პრობლემა გადავწყვიტეთ XGBoost-ით — gradient boosting-ის ოჯახის მოდელი, კარგად მუშაობს განსაკუთრებით დიდი feature სივრცის და კლასური დისბალანსის შემთხვევაში. მუშაობა დავყავით ეტაპებად: Cleaning → Feature Engineering → Feature Selection → Hyperparameter Tuning.

## რეპოზიტორიის სტრუქტურა

```
ieee-fraud-detection/
├── model_experiment_{მოდელის არქიტექტურა}.ipynb.ipynb
├── model_inference.ipynb
└── README.md
```

---

## Cleaning

### NaN ანალიზი

პირველ რიგში შევხედეთ, თუ რამდენი გამოტოვებული მნიშვნელობა აქვს თითოეულ სვეტს. დავყავით ოთხ ჯგუფად:

```
NaN = 0%      →  50 შევინახეთ
NaN 0–30%     →  80  შევავსეთ
NaN 30–70%    →  60  შევავსეთ
NaN 70–100%   → 241  წავშალეთ
```
):


### Quasi-Constant სვეტები

ცალკე შევხედეთ სვეტებს, სადაც ერთი მნიშვნელობა 99%+ შემთხვევაში მეორდება. ასეთ სვეტს პრაქტიკულად არ შეუძლია სასარგებლო ინფორმაციის გადმოცემა — მოდელი მათგან ვერ ისწავლის. ყველა ასეთი სვეტები ამოვიღეთ.

### NaN შევსება

კატეგორიული სვეტები: NaN შეიცვალა `'Unknown'`-ით — ცალკე კატეგორიად, რადგან "ინფორმაციის არარსებობა" თავად შეიძლება იყოს სიგნალი.

რიცხვითი სვეტები: NaN შეიცვალა median-ით. საშუალოსგან განსხვავებით, median გამძლეა outlier-ების მიმართ — ამ dataset-ში კი ტრანზაქციის თანხებს ბევრი ასეთი მნიშვნელობა აქვს.

---

## Feature Engineering

### TransactionAmt

ტრანზაქციის თანხა ძლიერ skewed განაწილებას ავლენდა — ბევრი პატარა თანხა და რამდენიმე ძალიან დიდი. ამის გამო log transform გამოვიყენეთ.

```python
TransactionAmt_log      = log(1 + TransactionAmt)
TransactionAmt_decimal  = TransactionAmt % 1
TransactionAmt_isround  = 1 თუ TransactionAmt მრგვალია, სხვაგვარად 0
```
<img width="1161" height="321" alt="image" src="https://github.com/user-attachments/assets/6a2d2647-d361-4005-983e-60d88a6a03d2" />



მარცხნივ ორიგინალი განაწილება (ძლიერ skewed), შუაში log-გარდაქმნის შემდეგ (ბევრად უფრო ნორმალური), მარჯვნივ fraud vs not-fraud შედარება. ჩანს, რომ fraud-ური ტრანზაქციები სხვა თანხებზეა კონცენტრირებული.

ათწილადი ნაწილი საინტერესო feature-ი გამოდგა: ჩვეულებრივი ტრანზაქციები ხშირად ზუსტი (მრგვალი) თანხებია, fraud-ული კი — ნაკლებად.

### დროის Feature-ები

TransactionDT Unix timestamp-ია. ამოვყავით:

```
Transaction_hour     = (TransactionDT // 3600) % 24
Transaction_day      = (TransactionDT // 86400) % 7
Transaction_week     = (TransactionDT // 604800) % 4
Is_night_fraud_hour  = 1 თუ საათი 5–10 შორისაა
```

<img width="1178" height="333" alt="image" src="https://github.com/user-attachments/assets/0e8f4306-5d4d-41e2-8884-b934b109f363" />


EDA-ში გამოჩნდა, რომ fraud rate განსაკუთრებით საღამოს 5–10 საათზე მაღლდება. ეს ბინარული feature-ად გადავიყვანეთ.

### Frequency Encoding

card1-ს 13,553 უნიკალური მნიშვნელობა ჰქონდა. Label Encoding ამ შემთხვევაში ცუდი არჩევანი იქნებოდა — ნომრები თვითნებური იქნებოდა. ამიტომ:

- `card1_freq` — რამდენჯერ გვხვდება ეს ბარათი dataset-ში
- `card1_fraud_rate` — ამ ბარათის ტრანზაქციების fraud rate (target encoding)
- `card2_freq`, `card5_freq` — ანალოგიურად

### კომბინირებული Feature-ები

- `card1_addr1_freq` — ბარათი + მისამართის კომბინაციის სიხშირე. ერთი ბარათი ბევრი სხვადასხვა მისამართიდან — fraud-ის სიგნალი.
- `card1_email_freq` — ბარათი + email domain კომბინაცია

### Email Domain

EDA-ში გამოჩნდა, რომ გარკვეული email domain-ები გაცილებით მაღალ fraud rate-ს ავლენდნენ:

<img width="431" height="336" alt="image" src="https://github.com/user-attachments/assets/88170b1a-2f7c-4eb5-a982-e068c8002089" />


```
protonmail.com  →  ~40% fraud rate
mail.com        →  ~18%
outlook.es      →  ~13%
gmail.com       →   ~2%
```

ამის საფუძველზე შევქმენით ბინარული feature `P_email_is_high_fraud`.

### კატეგორიული კოდირება

მცირე cardinality-ის კოლონებისთვის (`ProductCD`, `card4`, `card6`, `M1`–`M9`) გამოვიყენეთ Label Encoding. XGBoost-ისთვის ეს საკმარისია, რადგან ხის სტრუქტურა კარგად ართმევს თავს ნომინალურ მნიშვნელობებს.

---

## Feature Selection

### V კოლონების პრობლემა

339 V კოლონიდან ბევრი ძლიერ კორელაციაშია ერთმანეთთან. 0.95+ კორელაციის მქონე წყვილი ბევრია — ეს redundancy-ია, ანუ ისინი ერთსა და იმავე ინფორმაციას ატარებენ.

<img width="1305" height="738" alt="image" src="https://github.com/user-attachments/assets/420120df-7a86-407c-a2bf-edb8a9f7e0ae" />
<img width="1344" height="760" alt="image" src="https://github.com/user-attachments/assets/5c6c3606-0a89-4fb3-9ce0-05134c462f06" />


### v1 — XGBoost Importance > 0

სწრაფი XGBoost მოდელი (100 ხე) გავუშვით ყველა feature-ზე. შევინახეთ მხოლოდ ის feature-ები, რომლებიც მოდელმა საერთოდ გამოიყენა.

შედეგი: ~152 feature

### v2 — Decorrelation + Importance > 0

კორელირებული V კოლონების წყვილებიდან ვტოვებდით მაღალი importance-ის მქონეს, დანარჩენს ვშლიდით (ზღვარი: 0.95).

შედეგი: ნაკლები feature, ნაკლები redundancy

ორივე ვარიანტი 3-fold CV-ით შევაფასეთ. გამარჯვებული კოდში ავტომატურად შეირჩა.


TOP-30 feature-ების importance-ი. `card1_fraud_rate` და `TransactionAmt_log` ყველაზე მნიშვნელოვანი feature-ები გამოდგა, რაც ლოგიკურია — ბარათის ისტორიული fraud rate პირდაპირ კავშირშია თაღლითობის ალბათობასთან.

---

## Training

სულ 13 ექსპერიმენტი ჩავატარეთ. ყველა run MLflow-ზეა დალოგილი `XGBoost_Training` ექსპერიმენტში.

### Underfitting

| Run | პარამეტრები | Val AUC | მიზეზი |
|-----|------------|---------|--------|
| XGB_1 | depth=2, n=10, lr=0.3 | დაბალი | მოდელი ძალიან მარტივია |
| XGB_2 | depth=3, n=50, lr=0.001 | დაბალი | lr ძალიან პატარა, 50 ხე კმარა არ არის |

depth=2 ნიშნავს მაქსიმუმ 4 ფოთოლი ერთ ხეში — 150+ feature-ის მქონე პრობლემისთვის ეს საკმარისი არ არის. lr=0.001 კი ნიშნავს, რომ 50 ნაბიჯი თითქმის ვერ ცვლის მოდელს.

### Overfitting

| Run | პარამეტრები | Train AUC | Val AUC | Gap |
|-----|------------|-----------|---------|-----|
| XGB_3 | depth=15, n=1000, reg=0 | ~1.0 | დაბალი | >0.15 |
| XGB_4 | depth=10, n=500, alpha=0 | მაღალი | საშ. | >0.08 |

depth=15 ნიშნავს, რომ ერთ ხეს 32,768 ფოთოლი შეუძლია ჰქონდეს. 1000 ასეთი ხე train set-ს ფაქტიურად ზეპირობს. regularization-ის გარეშე (alpha=0, lambda=0) ეს კიდევ უფრო მძაფრდება.

### კარგი ბალანსი

| Run | ცვლილება | Val AUC | Gap | კომენტარი |
|-----|---------|---------|-----|-----------|
| XGB_7 | alpha=2, lambda=10 | საშ. | მცირე | ზედმეტი reg → underfit |
| XGB_8 | scale_pos_weight | კარგი | მცირე | imbalance-ს ეხმარება |
| XGB_9 | depth=4, n=800 | კარგი | მცირე | კარგი depth/estimators ბალანსი |
| XGB_10 | early_stopping=30 | კარგი | მცირე | ავტომატური გაჩერება |
| XGB_12 | ყველა გაერთიანებული | საუკეთესო | მცირე | — |





### საბოლოო მოდელი

```python
XGBClassifier(
    n_estimators     = 2000,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    scale_pos_weight = 29,
    tree_method      = 'hist'
)
```

პარამეტრების არჩევის ლოგიკა:

- `depth=6` — XGB_3-ის (depth=15) გამოცდილებიდან. 6 საკმარისი სირთულეა overfitting-ის გარეშე.
- `lr=0.05` + `n=2000` — XGB_6-ში (lr=0.01, n=300) ვნახეთ, რომ 300 ხე კმარა არ იყო. 2000 ხე + early stopping → ოპტიმალური გაჩერება.
- `subsample=0.8`, `colsample=0.8` — თითოეული ხე სხვადასხვა feature/sample ქვეჯგუფს ხედავს, overfitting-ს ამცირებს.
- `min_child_weight=5` — ფოთოლი მინიმუმ 5 sample-ს უნდა შეიცავდეს. XGB_3-ში (mcw=1) ძალიან მცირე ფოთლები ჩნდებოდა.
- `scale_pos_weight=29` — fraud კლასის 1:29 დისბალანსის კომპენსაცია. XGB_8-ის გამოცდილებიდან.

საბოლოო შედეგები:

```
CV Val AUC (3-fold):   ~0.92
Kaggle Public Score:    0.8239
Kaggle Private Score:   0.7588
```

Public და Private score-ს შორის ~0.065 სხვაობა მიუთითებს, რომ მოდელი გარკვეულწილად overfitted-ია public test set-ზე. ამის სავარაუდო მიზეზი — ჩვენი random CV split-ი არ ითვალისწინებს მონაცემების დროით სტრუქტურას. fraud detection-ში pattern-ები დროთა განმავლობაში იცვლება, ამიტომ მოდელი უნდა ვალიდირდეს "მომავალ" პერიოდზე და არა random sample-ზე.

---

## MLflow Tracking

MLflow ექსპერიმენტების ბმული: [https://dagshub.com/tgura23/ieee-fraud-detection](https://dagshub.com/tgura23/ieee-fraud-detection/experiments)

### ექსპერიმენტის სტრუქტურა

```
XGBoost_Training/
├── XGBoost_Cleaning
├── XGBoost_Feature_Engineering
├── XGBoost_Feature_Selection_v1_importance
├── XGBoost_Feature_Selection_v2_decorrelated
├── XGB_1_Underfit_shallow
├── XGB_2_Underfit_lowLR
├── XGB_3_Overfit_deep
├── XGB_4_Overfit_noreg
├── XGB_5_HighLR
├── XGB_6_LowLR
├── XGB_7_StrongReg
├── XGB_8_Imbalanced_spw
├── XGB_9_ShallowMany
├── XGB_10_EarlyStopping
├── XGB_11_LowColsample
├── XGB_12_Balanced
├── XGB_13_BEST_FullData
├── XGBoost_Inference_TestSet
└── XGBoost_Submission_Log
```

### ლოგირებული მეტრიკები

| მეტრიკა | აღწერა |
|---------|--------|
| `mean_train_auc` | Train fold-ების საშუალო AUC |
| `mean_val_auc` | Validation fold-ების საშუალო AUC |
| `overfit_gap` | Train − Val AUC სხვაობა |
| `std_val_auc` | Val AUC-ის სტანდარტული გადახრა |
| `n_features` | გამოყენებული feature-ების რაოდენობა |
| `n_samples` | სასწავლო სტრიქონების რაოდენობა |
| `dropped_high_nan` | წაშლილი NaN კოლონების რაოდენობა |
| `mean_best_iter` | Early stopping-ის საშუალო გაჩერების წერტილი |

### Model Registry

საბოლოო Pipeline (`SimpleImputer` + `XGBClassifier`) დარეგისტრირებულია Model Registry-ში სახელით `XGBoost_Fraud_Pipeline`:

```python
pipeline = mlflow.sklearn.load_model("models:/XGBoost_Fraud_Pipeline/Production")
predictions = pipeline.predict_proba(X_test)[:, 1]
```

---

## გაუმჯობესების მიმართულებები

**Time-based CV split** — ამჟამინდელი random split-ი არ ითვალისწინებს დროით სტრუქტურას. fraud pattern-ები იცვლება და მოდელი უნდა ვალიდირდეს მომავალ პერიოდზე. ეს public/private gap-ს შეამცირებს.

**card1_fraud_rate test-ზე** — ამჟამად test set-ისთვის ეს feature 0-ია, რადგან test-ს isFraud არ აქვს. სწორი მიდგომა: train-ზე დათვლილი fraud rates-ის შენახვა და test-ზე apply.

**LightGBM / CatBoost** — ამ კომპეტიციაში top solution-ების უმეტესობა სწორედ ამ ორ მოდელზე ან მათ ensemble-ზე იყო დაფუძვნებული.

**Ensemble** — XGBoost + LightGBM + CatBoost-ის შეჯამება averaging-ით ჩვეულებრივ +0.02–0.03 AUC-ს იძლევა.     + სხვა მოდელები არ მაქვს განხილული, რაც ამცირებს შედეგს.
<img width="1514" height="275" alt="image" src="https://github.com/user-attachments/assets/e72886e7-2968-43c6-860f-1f564d3b23ed" />

