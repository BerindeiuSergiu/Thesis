# Fast3R Indoor Reconstruction - Aplicatie practica licenta

## Repository

Codul sursa al aplicatiei este disponibil public la:

https://github.com/BerindeiuSergiu/Thesis

Repository-ul contine codul sursa al aplicatiei desktop, pipeline-ul de integrare Fast3R, codul Python pentru experimente si workflow-ul de training/evaluare Vast.ai. Fisierele binare compilate, codul vendorizat al modelului Fast3R, documentele lucrarii, seturile de date, modelele, arhivele si rezultatele generate local nu sunt incluse in repository.

## Credit Fast3R

Aplicatia utilizeaza modelul si codul Fast3R ca dependinta externa:

- repository oficial Fast3R: https://github.com/facebookresearch/fast3r
- lucrare: `Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass`, CVPR 2025
- model pretrained folosit implicit: https://huggingface.co/jedyang97/Fast3R_ViT_Large_512
- licenta Fast3R upstream: FAIR Noncommercial Research License

Codul Fast3R upstream nu este vendorizat in acest repository de predare; trebuie instalat separat conform pasilor de mai jos.

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

2. Instalati Python daca nu este deja disponibil.

Versiunea recomandata este Python 3.12. Aplicatia poate rula si cu versiuni Python 3.10-3.12, in functie de compatibilitatea pachetelor PyTorch/Fast3R instalate local.

Pe Windows, instalati Python din Microsoft Store, de pe https://www.python.org/downloads/ sau prin `winget`:

```powershell
winget install Python.Python.3.12
python --version
```

Pe Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 --version
```

3. Creati si activati un mediu virtual Python.

Daca modulul `venv` nu este disponibil pe Linux, instalati-l rapid:

```bash
sudo apt install python3-venv
```

Creare mediu virtual:

```bash
python -m venv .venv
```

Daca pe Linux comanda `python` nu este disponibila, folositi:

```bash
python3 -m venv .venv
```

Activare pe Windows:

```bash
.venv\Scripts\activate
```

Activare pe Linux/macOS/Git Bash:

```bash
source .venv/bin/activate
```

4. Instalati dependintele aplicatiei:

```bash
python -m pip install --upgrade pip
python -m pip install -r src/requirements.txt
```

Fisierul `src/requirements.txt` este lista de dependinte pentru aplicatia desktop. Fisierul `requirements.txt` din radacina repository-ului este pastrat pentru scripturi si experimente vechi, iar `experiments/vast_training/requirements-lora.txt` este folosit doar pentru etapa optionala de LoRA/fine-tuning.

5. Instalati separat dependinta Fast3R folosita de pipeline. Varianta folosita in proiect este repository-ul upstream Fast3R; acesta poate fi clonat separat si instalat in acelasi mediu virtual conform instructiunilor upstream.

Exemplu local, din afara repository-ului predat:

```bash
git clone https://github.com/facebookresearch/fast3r.git fast3r_external
python -m pip install -r fast3r_external/requirements.txt
python -m pip install -e fast3r_external
```

Modelul implicit `jedyang97/Fast3R_ViT_Large_512` este configurat in `src/config/settings.yaml` si se descarca prin Hugging Face la prima utilizare, daca nu exista deja in cache-ul local.

Pentru profilul `Fine-Tuned Fast3R`, copiati exportul local al modelului in folderul aplicatiei:

```text
src/data/models/fast3r_arkitscenes_ga_head_hf/
  config.json
  model.safetensors
  README.md
```

Aceste weight-uri nu sunt incluse in repository deoarece `model.safetensors` este un fisier binar mare.

## Pasi de lansare

Din radacina repository-ului:

```bash
python -m src.main
```

Alternativ, folositi scriptul shell inclus:

```bash
./run_app.sh
```

Aplicatia porneste interfata desktop PyQt6. Utilizatorul creeaza o scena, selecteaza un video monocular indoor, alege profilul de model si presetul de reconstructie, apoi ruleaza pipeline-ul local. Rezultatele sunt generate local in `src/outputs/`, folder ignorat de Git.
