# Fast3R Indoor Reconstruction - Aplicatie practica licenta

## Repository

Codul sursa al aplicatiei este disponibil public la:

https://github.com/BerindeiuSergiu/Thesis

Repository-ul contine codul sursa al aplicatiei desktop, pipeline-ul de integrare Fast3R, codul Python pentru experimente si workflow-ul de training/evaluare Vast.ai. Fisierele binare compilate, codul vendorizat al modelului Fast3R, documentele lucrarii, seturile de date, modelele, arhivele si rezultatele generate local nu sunt incluse in repository.

## Pasi de compilare

Aplicatia este o aplicatie Python si nu necesita compilare intr-un executabil pentru rularea din sursa.

Verificare optionala a codului Python:

```bash
python -m compileall src experiments scripts
```

## Pasi de instalare

1. Clonati repository-ul:

```bash
git clone https://github.com/BerindeiuSergiu/Thesis.git
cd Thesis
```

2. Creati si activati un mediu virtual Python:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Instalati dependintele aplicatiei:

```bash
python -m pip install --upgrade pip
python -m pip install -r src/requirements.txt
```

4. Instalati separat dependinta Fast3R folosita de pipeline. Varianta folosita in proiect este repository-ul upstream Fast3R; acesta poate fi clonat separat si instalat in acelasi mediu virtual conform instructiunilor upstream.

Exemplu local, din afara repository-ului predat:

```bash
git clone https://github.com/jedyang97/Fast3R.git fast3r_external
python -m pip install -r fast3r_external/requirements.txt
python -m pip install -e fast3r_external
```

Modelul implicit `jedyang97/Fast3R_ViT_Large_512` este configurat in `src/config/settings.yaml` si se descarca prin Hugging Face la prima utilizare, daca nu exista deja in cache-ul local.

## Pasi de lansare

Din radacina repository-ului:

```bash
python -m src.main
```

Aplicatia porneste interfata desktop PyQt6. Utilizatorul creeaza o scena, selecteaza un video monocular indoor, alege profilul de model si presetul de reconstructie, apoi ruleaza pipeline-ul local. Rezultatele sunt generate local in `src/outputs/`, folder ignorat de Git.
