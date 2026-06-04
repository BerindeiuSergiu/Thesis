# THESIS MASTER KNOWLEDGE

Acest fișier este depozitul intern principal pentru redactarea tezei. El consolidează informația validată tehnic din:

- `THESIS_KNOWLEDGE_BASE.md`
- `THESIS_INTERNAL_FILE_LEGEND.md`
- `Documentatie Proiect (1).docx`
- artefactele și codul efectiv din repository

Scopul lui este dublu:

1. să păstreze afirmațiile tehnice care pot fi susținute direct de dovezile locale;
2. să sincronizeze redactarea din `THESIS_DRAFT.md` cu rezultatele, limitările și deciziile de inginerie reale.

Acest document îl înlocuiește practic, pentru fluxul curent de scriere, pe vechiul `THESIS_KNOWLEDGE_BASE.md` atunci când apar contradicții. În special, secțiunea de fine-tuning trebuie interpretată după rezultatele mai noi din `agent/CODEX_WORKLOG.md` și din helper-ele de redactare dedicate, nu după notele mai vechi care presupuneau folosirea split-ului oficial ARKitScenes Validation.

## 1. Stil și reguli de redactare

### 1.1. Stilul țintă

Din analiza documentului de referință `Documentatie Proiect (1).docx` rezultă următoarele caracteristici stilistice care trebuie păstrate în teza finală:

- ton formal, academic și explicit;
- paragrafe compacte, nu listări excesive;
- introducerea mai întâi a contextului conceptual, apoi a detaliilor de implementare;
- tranziții clare între capitole și subcapitole;
- formulări de tipul „În acest context…”, „În cadrul proiectului…”, „Astfel…”, „Prin această abordare…”;
- referințe naturale la figuri și tabele în corpul textului;
- accent pe justificarea deciziilor tehnice, nu doar pe enumerarea lor.

### 1.2. Reguli ferme pentru draft

- Teza este redactată integral în limba română.
- Numele de tehnologii, biblioteci și modele pot rămâne în engleză.
- Nu se documentează scripturile de execuție, shell-urile sau automatizările operaționale.
- Nu se inventează rezultate, metrici sau experimente.
- Dacă o informație rămâne incertă, ea se mută în `THESIS_GAPS.md`, nu în corpul principal al tezei.

## 2. Identitatea proiectului

### 2.1. Titlu de lucru recomandat

Titlul de lucru cel mai fidel proiectului este:

**Reconstrucția 3D a scenelor interioare din video monocular folosind Fast3R, selecție de cadre orientată geometric și ieșiri duale Gaussian/mesh**

### 2.2. Ce este proiectul în esență

Proiectul este o aplicație desktop locală pentru reconstrucție 3D indoor din videoclip, construită în jurul modelului Fast3R și extinsă prin:

- selecție de cadre bazată pe scoruri vizuale și pe un pas de probă geometric;
- filtrare și retenție a punctelor în funcție de încredere;
- normalizare globală a scalei în metri;
- reducerea timpului până la obținerea unei scene inspectabile;
- ramificare din același nor de puncte către două ieșiri:
  - reprezentare de tip Gaussian splat;
  - reconstrucție mesh.

### 2.3. Diferențiatori reali ai proiectului

Contribuția practică nu este „folosirea Fast3R ca atare”, ci integrarea lui într-un pipeline inginerește coerent:

- intrare video, nu colecție manuală de imagini;
- selecție de cadre care încearcă să maximizeze utilitatea geometrică;
- postprocesare și retenție orientate pe robustețe;
- normalizare de scală pentru utilizare mai practică;
- generare dintr-o singură reconstrucție comună a două produse finale complementare;
- accent explicit pe performanță și pe reducerea timpului de așteptare pentru utilizator.

### 2.4. Performanța ca obiectiv central

Unul dintre cele mai puternice argumente ale aplicației este reducerea timpului până la obținerea unei scene 3D inspectabile. Fluxurile clasice bazate pe etape iterative, precum COLMAP, și fluxurile care optimizează separat o reprezentare Gaussian splat pot introduce timpi de așteptare semnificativi. Aplicația folosește inferența multi-view Fast3R, selecția cadrelor reprezentative și o geometrie procesată comună pentru exporturile Gaussian splat și mesh.

Rezultatele locale înregistrate pentru rulări cu export Gaussian includ aproximativ `148 s` pentru `80` cadre, `224 s` pentru `120` cadre și `487 s` pentru `300` cadre. Ca reper istoric intern, un experiment COLMAP sparse SfM arhivat a raportat aproximativ `8 minute` pentru `100` cadre eșantionate și a reconstruit `24` poziții de cameră cu `1.230` puncte sparse.

Acest reper COLMAP nu reprezintă un benchmark controlat direct împotriva COLMAP dense MVS sau a unor aplicații 3DGS complet optimizate. Afirmația corectă este că performanța și feedback-ul rapid sunt obiective centrale ale proiectului, iar rezultatele locale susțin această direcție. O comparație cross-tool controlată rămâne o extensie importantă.

#### Performanța ca o contribuție primară a aplicației

În redactarea tezei, performanța nu trebuie tratată ca o metrică secundară plasată doar în capitolul de rezultate. Ea este una dintre contribuțiile principale ale aplicației, deoarece întregul flux software a fost construit în jurul reducerii timpului de așteptare dintre încărcarea unui videoclip și obținerea unei scene 3D care poate fi inspectată de utilizator.

Această idee trebuie reluată în Capitolul 1 ca motivație, în Capitolul 5 ca decizie de arhitectură și în Capitolul 6 ca rezultat experimental. Formularea centrală recomandată este:

> Pe lângă calitatea geometrică a reconstrucției, proiectul urmărește reducerea timpului până la obținerea unei scene inspectabile. Din acest motiv, performanța este tratată ca o cerință funcțională a aplicației, nu doar ca o observație experimentală.

Elementele care trebuie accentuate:

- aplicația este construită pentru a reduce timpul până când utilizatorul poate inspecta o scenă 3D;
- Fast3R a fost ales nu doar pentru calitatea reconstrucției, ci și pentru faptul că permite inferență multiview într-un singur forward pass, evitând multe etape iterative;
- selecția cadrelor reduce costul înainte de inferență, deoarece nu toate cadrele dintr-un video lung contribuie în mod egal la geometrie;
- ramurile Gaussian și mesh pornesc din același nor de puncte procesat, ceea ce evită repetarea reconstrucției pentru fiecare format de ieșire;
- metricul practic al aplicației nu este doar acuratețea, ci și `time-to-inspectable-scene`;
- limitările de performanță sunt vizibile în eșecurile reale, mai ales în rulările cu multe puncte, setări high-detail sau depășiri de memorie GPU.

Această secțiune trebuie să facă diferența dintre proiect și o simplă demonstrație de model. Modelul este important, dar valoarea practică a proiectului vine din felul în care modelul este integrat într-un flux rapid, local și reutilizabil.

Metrici care trebuie folosite pentru susținerea acestei contribuții:

| Dovadă locală | Valoare utilă în teză | Interpretare |
| --- | --- | --- |
| Fast3R, rulare cu export Gaussian | aproximativ `148 s` pentru `79` cadre și `6,306,527` puncte brute | rezultat inspectabil în câteva minute |
| Fast3R, rulare cu export Gaussian | aproximativ `224 s` pentru `120` cadre și `9,631,862` puncte brute | creșterea numărului de cadre rămâne practică |
| Fast3R, rulare cu export Gaussian | aproximativ `487 s` pentru `300` cadre și `14,918,694` puncte brute | arată limita superioară testată local pentru rulări reușite |
| Scenă reprezentativă a aplicației | `180` cadre, `26,264,029` puncte brute, `8,146,338` puncte finale/Gaussians | demonstrează volum mare de geometrie procesată în fluxul aplicației |
| COLMAP sparse SfM arhivat | aproximativ `8 minute`, `100` cadre, `1,230` puncte sparse, `24` camere recuperate | reper istoric: geometrie curată, dar sparse și lentă pentru feedback dens |
| DUSt3R standalone experimental | `1604.15 s` pentru `40` cadre și `335,133` puncte | justifică renunțarea la ruta standalone în favoarea Fast3R |
| Rulare high-detail eșuată | aproximativ `30,196,929` puncte brute și eroare de alocare memorie | arată că performanța și memoria sunt constrângeri reale de proiectare |

Afirmația de redactare trebuie să fie puternică, dar precisă: aplicația nu promite reconstrucție în timp real și nu înlocuiește fluxurile 3DGS complet optimizate. Contribuția este reducerea timpului până la primul rezultat 3D inspectabil și reutilizabil, printr-un pipeline integrat.

### 2.5. Poziționarea aplicației față de alte fluxuri software

Aplicația trebuie prezentată ca o contribuție software proprie, nu doar ca un wrapper minimal pentru model. Ea integrează într-un singur flux local selecția cadrelor, inferența Fast3R, filtrarea, normalizarea scalei, generarea inițializării Gaussian, exportul mesh, istoricul scenelor și vizualizarea rezultatului. Avantajul urmărit este reducerea numărului de etape separate și a timpului până la primul rezultat inspectabil.

Pentru comparația conceptuală sunt relevante următoarele fluxuri:

| Flux software | Etape suplimentare înaintea rezultatului final | Nivelul dovezii |
| --- | --- | --- |
| COLMAP SfM / MVS | Extracție de trăsături, matching, reconstrucție sparse, undistortion, stereo, fusion și mesh opțional. | Documentație oficială și reper intern sparse SfM de aproximativ `8 minute` pentru `100` cadre. |
| Nerfstudio Splatfacto | Procesarea video-ului folosește COLMAP și FFmpeg, apoi urmează antrenarea Splatfacto și exportul splat-ului antrenat. | Documentație oficială; fără benchmark local. |
| Implementarea INRIA 3DGS | Pregătire a intrării SfM de tip COLMAP, urmată de optimizare Gaussian iterativă; configurația de referință folosește implicit `30.000` iterații. | Implementare oficială; fără benchmark local. |
| Jawset Postshot | Import video sau imagini, camera tracking multi-step pentru poziții și puncte sparse atunci când acestea lipsesc, apoi antrenarea radiance field-ului. | Documentație oficială; fără benchmark local. |
| Aplicația proiectului | Selecție de cadre, inferență Fast3R, procesare geometrică comună și export Gaussian/mesh din același rezultat. | Timpi locali măsurați pentru fluxul implementat. |

Comparația trebuie formulată cu grijă. Fluxurile 3DGS complete rezolvă o sarcină mai amplă deoarece optimizează separat o reprezentare radiance-field antrenată. Aplicația proiectului prioritizează obținerea rapidă a unei inițializări Gaussian inspectabile și a unui mesh, nu înlocuirea tuturor funcțiilor oferite de aceste instrumente.

Surse oficiale utile pentru redactare:

- [COLMAP tutorial](https://colmap.github.io/tutorial.html)
- [Nerfstudio custom-data guide](https://docs.nerf.studio/quickstart/custom_dataset.html)
- [Nerfstudio Splatfacto guide](https://docs.nerf.studio/nerfology/methods/splat.html)
- [INRIA Gaussian Splatting reference implementation](https://github.com/graphdeco-inria/gaussian-splatting)
- [Postshot importing guide](https://www.jawset.com/docs/d/Postshot%2BUser%2BGuide/Importing%2BImages)
- [Postshot training configuration](https://activation.jawset.com/docs/d/Postshot%2BUser%2BGuide/Interface/Training%2BConfiguration)

## 3. Arhitectura sistemului

### 3.1. Formă de aplicație

- aplicație desktop locală;
- interfață PyQt6;
- execuție a pipeline-ului într-un worker separat față de UI;
- istoric local de scene;
- vizualizare locală a rezultatelor prin viewer dedicat.

### 3.2. Module funcționale majore

Modulele importante care trebuie descrise la nivel conceptual în teză sunt:

- interfața grafică de configurare și control;
- worker-ul care rulează reconstrucția în fundal;
- orchestratorul principal al pipeline-ului;
- procesarea video și extragerea cadrelor;
- selecția de cadre;
- inferența Fast3R și estimarea pozițiilor camerelor;
- normalizarea scalei;
- filtrarea și subeșantionarea norului de puncte;
- ramura Gaussian splat;
- ramura mesh;
- viewer-ul de scene rezultate.

### 3.3. Fluxul logic de date

Fluxul corect pentru descriere este:

1. selectarea videoclipului;
2. analizarea și eșantionarea cadrelor candidate;
3. selecția finală de cadre;
4. inferență Fast3R pe setul selectat;
5. extragerea norului de puncte și a pozițiilor camerelor;
6. filtrare numerică și retenție pe bază de încredere;
7. normalizare de scală;
8. postprocesare comună a norului de puncte;
9. generare Gaussian splat și/sau mesh;
10. salvare artefacte și metadate;
11. vizualizare locală.

## 4. Tehnologii folosite și justificarea lor

### 4.1. Fast3R

Fast3R este modelul central al proiectului. Repository-ul local confirmă că este implementarea oficială a lucrării:

- *Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass*,
- CVPR 2025,
- autori: Jianing Yang, Alexander Sax, Kevin J. Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, Matt Feiszli.

Rolul lui în proiect:

- reconstrucție multiview densă;
- predicție de puncte 3D și semnale de încredere;
- estimare ulterioară a pozițiilor camerelor;
- model de bază și pentru experimentul de fine-tuning.

### 4.2. Open3D

Open3D este baza operațiilor geometrice:

- citire și scriere de nori de puncte și mesh-uri;
- estimare de normale;
- reconstrucție Poisson;
- măsurători geometrice și verificări de manifold / watertightness;
- suport practic pentru viewer-ul local și postprocesare.

### 4.3. PyQt6

PyQt6 oferă interfața de utilizare:

- configurare parametri;
- inițiere rulare;
- feedback de progres;
- listă de scene salvate;
- pornirea vizualizatorului.

### 4.4. PyTorch

PyTorch este folosit pentru:

- încărcarea și rularea modelului Fast3R;
- inferență pe GPU;
- infrastructura de fine-tuning și evaluare.

## 5. Date, preprocesare și reprezentarea rezultatelor

### 5.1. Tipurile de date folosite în proiect

Există două familii de date distincte și trebuie separate clar în teză:

1. datele aplicației:
   - videoclipuri reale indoor capturate manual;
2. datele pentru adaptarea modelului:
   - ARKitScenes, procesat pentru antrenare și evaluare.

### 5.2. Datele aplicației

Pipeline-ul principal nu pornește de la un dataset etichetat, ci de la video-uri reale cu:

- mișcare de cameră neuniformă;
- cadre redundante;
- blur;
- zone slab texturate;
- variații de iluminare.

Rezultatul fiecărei rulări este salvat într-un director de scenă și include:

- metadate generale;
- raport de retenție a punctelor;
- raport de normalizare a scalei;
- raport de procesare a norului de puncte;
- artefacte Gaussian splat;
- artefacte mesh, dacă ramura mesh este activată.

### 5.3. Preprocesarea cadrelor

Selecția de cadre combină două niveluri:

- shortlist vizual construit din metrice de imagine;
- rafinare geometrică bazată pe un pas de probă Fast3R la rezoluție redusă.

Indicatorii vizuali utilizați includ:

- claritate prin variația Laplacianului;
- entropie;
- expunere și clipping;
- textură bazată pe ORB;
- deduplicare prin similaritate.

### 5.4. Retenția punctelor și curățarea geometrică

După inferență, proiectul nu folosește direct ieșirea brută a modelului. Sunt aplicate:

- prag minim de încredere;
- păstrare parțială pe cuantile de încredere;
- eliminarea unor vederi slabe prin praguri pe statistici per-view;
- filtrare numerică;
- eliminarea outlier-ilor;
- voxel downsampling.

### 5.5. Normalizarea scalei

Normalizarea scalei este o etapă distinctă și importantă. Ea încearcă mai întâi să folosească informație explicită din metadate, iar când aceasta nu este disponibilă recurge la o referință geometrică din scenă. În rularea reprezentativă păstrată local, factorul de scalare a fost estimat din înălțimea camerei/încăperii și a avut valoarea:

- `scale_to_meters = 7.365792240649551`

## 6. Rezultate experimentale validate

### 6.1. Rulare reprezentativă a aplicației

Rularea locală reprezentativă care merită descrisă în capitolul de rezultate este cea care a generat scena completă cu Gaussian splat și mesh. Valorile validate sunt:

- cadre totale în video: `5373`;
- cadre candidate analizate: `538`;
- shortlist vizual: `522`;
- cadre selectate final: `180`;
- puncte 3D brute extrase după pruning per-view: `26,264,029`;
- puncte finale după filtrare și voxelizare: `8,146,338`.

Pentru aceeași scenă:

- număr final de Gaussiene: `8,146,338`;
- mesh final: `179,965` triunghiuri;
- mesh-ul nu este watertight;
- mesh-ul are `32` componente conexe după curățarea principală.

### 6.2. Exemple de timp de execuție

Din sumarul experimental local sunt susținute următoarele exemple de runtime:

- aproximativ `148.27 s` pentru o rulare de tip 80 cadre;
- aproximativ `223.68 s` pentru 120 cadre;
- aproximativ `487.16 s` pentru 300 cadre.

Din tabelul de timpi pe etape rezultă că inferența domină costul total, iar ramura Gaussian splat poate adăuga zeci de secunde la rulări dense.

Pentru context, un experiment intern arhivat bazat pe COLMAP sparse SfM a raportat aproximativ `8 minute` pentru `100` cadre eșantionate și a produs `1.230` puncte sparse din `24` poziții de cameră reconstruite. Reperul susține motivarea orientării către performanță, dar nu trebuie descris ca o comparație directă controlată cu COLMAP dense MVS sau cu aplicații 3DGS complet optimizate.

### 6.3. Exemple de eșecuri reale

Limitările operaționale validate local includ:

- erori de memorie GPU pentru unele configurări cu multe cadre sau detaliu ridicat;
- eșecuri de alocare RAM în postprocesare pentru nori foarte mari;
- mesh-uri neetanse și cu auto-intersecții pe scene dificile;
- dependența scalei de o referință geometrică atunci când metadatele absolute lipsesc.

## 7. Fine-tuning: afirmații permise și afirmații interzise

### 7.1. Ce s-a întâmplat sigur

Există suficiente dovezi locale pentru a afirma următoarele:

- a fost realizat un run de fine-tuning Fast3R pe ARKitScenes cu adaptare de tip head-only;
- setarea a vizat capul de predicție `downstream_head`, cu backbone înghețat;
- profilul principal a folosit:
  - `num_views = 6`
  - `image_size = 512`
  - `batch_size = 1`
  - `precision = bf16-mixed`
  - `lr = 1.5e-5`
  - subset de antrenare de `16000` eșantioane
  - subset de validare de `50` eșantioane
- în jurnalul local al run-ului principal există 8 puncte de validare;
- cel mai bun `val/loss` intern a fost:
  - `0.06770282983779907` la pasul `6999`;
- checkpoint-ul final al run-ului principal a avut:
  - `val/loss = 0.07549560815095901` la pasul `7999`.

### 7.2. Ce trebuie spus cu grijă despre split

Cea mai importantă precizare metodologică este aceasta:

- comparația finală pretrained versus fine-tuned care trebuie raportată în teză **nu** folosește benchmark-ul oficial ARKitScenes Validation în forma lui standard păstrată pe mașina finală;
- în urma problemelor de recovery de pe mașina remote, evaluarea a fost refăcută pe un **synthetic holdout** de 20 scene construit din scenele ARKitScenes Training.

Split-ul reconstruit care poate fi documentat este:

- `88` scene pentru partea de train;
- `20` scene pentru partea de test/holdout;
- `293,829` frame-uri în partea de train;
- `81,277` frame-uri în partea de holdout.

### 7.3. Rezultatul principal care poate fi raportat

Acesta este rezultatul central și sigur pentru teză:

- checkpoint-ul pretrained a obținut `val/loss ≈ 0.0845`;
- checkpoint-ul fine-tuned de referință (`v4`) a obținut `val/loss ≈ 0.07535`;
- îmbunătățirea este de ordinul a `10–11%` pe această măsură de validare.

Interpretarea permisă este:

- fine-tuning-ul a redus loss-ul de validare pe subsetul indoor reconstruit;
- rezultatul sugerează o adaptare utilă la domeniul țintă;
- afirmația rămâne una de tip pilot, nu benchmark oficial complet.

### 7.4. Rezultatele negative care trebuie păstrate

Teza trebuie să includă și faptul că „mai mult antrenament” nu a însemnat automat performanță mai bună.

Rezultatele validate sunt:

- reluarea antrenării după checkpoint-ul bun a degradat rezultatul la:
  - `val/loss ≈ 0.09779`;
- un branch ulterior cu mai multe date, evaluat extern, a produs:
  - pretrained `≈ 0.08281`
  - fine-tuned `≈ 0.08829`
  - deci mai slab decât baseline-ul pretrained pe același subset.

Concluzia permisă este:

- selecția checkpoint-ului este critică;
- continuarea antrenării după punctul bun poate degrada modelul;
- experimentele de continuare nu au depășit checkpoint-ul `v4`, care rămâne cel mai bun rezultat raportabil.

## 8. Afirmații care trebuie evitate în teză

Nu trebuie scris:

- că validarea finală a fost făcută pe protocolul oficial complet ARKitScenes Validation;
- că modelul fine-tuned este „mai bun în general”;
- că branch-urile de continuare au îmbunătățit clar modelul;
- că ieșirea Gaussian este rezultatul unui pipeline complet de 3D Gaussian Splatting antrenat separat;
- că aplicația este demonstrat mai rapidă decât orice pipeline COLMAP dense MVS sau orice aplicație 3DGS.

Se poate scrie:

- că modelul fine-tuned a obținut un loss de validare mai mic pe subsetul reconstruit;
- că rezultatul susține ipoteza unei adaptări utile la scene indoor;
- că experimentele ulterioare au arătat și riscul de supra-antrenare;
- că reducerea timpului până la obținerea unei scene inspectabile este un obiectiv central și unul dintre cele mai puternice argumente ale aplicației;
- că timpii locali Fast3R și reperul istoric COLMAP susțin motivarea unei evaluări orientate spre performanță.

## 9. Capitole și idei-cheie pentru draft

### Capitolul 1

- problema reconstrucției 3D indoor din video monocular;
- motivația pentru un pipeline practic, nu doar o demonstrație de inferență;
- reducerea timpului până la obținerea unei scene inspectabile drept obiectiv central și contribuție principală;
- performanța trebuie introdusă de aici ca problemă de utilizare: fluxurile lente rup bucla de feedback a utilizatorului;
- obiectivele duble: aplicație utilizabilă + investigație de adaptare a modelului.

### Capitolul 2

- Fast3R ca model central;
- noțiuni de reconstrucție multiview;
- selecția de cadre și redundanța temporală;
- normalizarea scalei;
- Poisson reconstruction și ieșiri Gaussian/mesh.

### Capitolul 3

- videoclipuri reale drept intrare operațională;
- ARKitScenes drept suport pentru adaptarea modelului;
- pipeline de selecție și curățare a datelor;
- synthetic holdout și limitarea lui metodologică.

### Capitolul 4

- arhitectura Fast3R folosită în proiect;
- head-only fine-tuning pe ARKitScenes;
- hiperparametri relevanți;
- interpretarea prudentă a rezultatelor și a degradărilor ulterioare.

### Capitolul 5

- descrierea aplicației desktop;
- responsabilitățile modulelor;
- modul în care modelul este integrat în flux;
- deciziile de arhitectură luate pentru performanță: selecție de cadre înainte de inferență, Fast3R ca forward-pass multiview, procesare comună a geometriei și ramificare Gaussian/mesh fără reconstrucție duplicată;
- structură de output și viewer.

Subcapitolul despre aplicație trebuie să fie tratat ca o contribuție software proprie. Nu este suficientă descrierea modelului Fast3R; trebuie explicat cum aplicația transformă modelul într-un flux utilizabil:

- workflow scene-first, în care utilizatorul creează o scenă cu nume, video, descriere, pipeline, profil de greutăți și preset;
- Scene Library cu scene locale, status și tip de reconstrucție;
- Reconstruction Properties cu input, pipeline, profil de greutăți, preset, status și acțiuni;
- profiluri de greutăți: `Default Fast3R` și `Fine-Tuned Fast3R`;
- rularea pipeline-ului într-un `QThread`, pentru ca interfața să nu fie blocată;
- tracker vizual pentru etapele Frames, Fast3R, Geometry și Output;
- viewer Viser încorporat prin `QWebEngineView`, fără deschiderea obligatorie a unui browser extern;
- outputuri salvate într-o structură de scenă reproductibilă, cu JSON-uri de metadate și rapoarte;
- meniurile `Workflow` și `View`, inclusiv restaurarea dock-urilor închise;
- `Open Output` disponibil permanent în afara rulărilor active, cu mesaj de ghidare dacă outputurile nu au fost încă construite.

Pentru acest capitol trebuie folosite explicit următoarele figuri:

- `[FIGURA 5.1 - Arhitectura aplicatiei desktop]`
- `[FIGURA 5.2 - Fluxul UI scene-first al aplicatiei]`
- `[FIGURA 5.3 - Structura outputului pentru o scena reconstruita]`
- `[FIGURA 5.4 - Interfata principala a aplicatiei desktop]`
- `[FIGURA 5.5 - Dialogul New Scene si selectia profilului de greutati]`
- `[FIGURA 5.6 - Scene Library si panoul Reconstruction Properties]`
- `[FIGURA 5.7 - Viewerul Viser incorporat pentru scena reconstruita]`

### Capitolul 6

- metrici și exemple concrete din scenele reconstruite;
- timpi, costuri și limitări, cu accent pe `time-to-inspectable-scene`;
- tabel și grafic pentru runtime în funcție de numărul de cadre;
- grafic stacked bar pentru timpii pe etape: selecție cadre, inferență, normalizare scală, procesare comună, ramură Gaussian, ramură mesh;
- comparația pretrained vs fine-tuned;
- exemple de succes și de eșec.

### Capitolul 7

- proiectul a produs un pipeline funcțional;
- fine-tuning-ul a oferit un câștig punctual, dar nu a fost robust la continuări;
- direcții viitoare: benchmark mai mare, mesh mai robust, scale mai bine ancorată, captură și evaluare mai controlate.

## 10. Figuri și tabele care merită prezente

### Figuri obligatorii

- arhitectura generală a sistemului;
- fluxul complet al pipeline-ului;
- interfața aplicației;
- viewer-ul scenei rezultate;
- exemplu de scenă reconstruită;
- runtime în funcție de numărul de cadre;
- timp pe etape pentru una sau mai multe rulări reprezentative;
- curbă de validare pentru run-ul de fine-tuning;
- comparație vizuală între output Gaussian și output mesh.

### Tabele obligatorii

- tehnologii folosite și rolul lor;
- parametrii principali ai pipeline-ului de producție;
- tabel de performanță: cadre, timp total, puncte brute, puncte procesate, status;
- tabel comparativ: COLMAP sparse, MiDaS, DUSt3R standalone, Fast3R application pipeline;
- sumarul scenei reprezentative;
- hiperparametrii de fine-tuning;
- comparația pretrained vs fine-tuned;
- sumarul limitărilor observate.

### Registru exact de figuri și tabele generate din date locale

Aceste marcaje trebuie folosite explicit în textul final al lucrării. Fiecare marcaj are un script corespunzător în `helper/script_for_plotting`, iar scripturile au fost rulate cu succes local. Pentru capturi vizuale ale outputului modelului se aplică o restricție separată: în forma curentă a tezei se vor insera doar outputuri vizuale Fast3R regenerate cu greutățile noi/fine-tuned; outputurile vechi pot fi folosite ca date de performanță și metadate, dar nu ca imagini finale ale modelului.

| Marcaj de inserat în teză | Capitol recomandat | Ce arată concret | Script reproductibil |
| --- | --- | --- | --- |
| `[FIGURA 1.1 - Contributiile principale ale lucrarii]` | Capitolul 1 | Contribuțiile: aplicație video-to-scene, selecție cadre, Fast3R, Gaussian/mesh, evaluare performanță, fine-tuning GA-head | `helper/script_for_plotting/figura_1_1_contributiile_principale_ale_lucrarii.py` |
| `[TABELUL 2.1 - Comparatia principalelor solutii analizate]` | Capitolul 2 | COLMAP, MiDaS-style depth, DUSt3R standalone și Fast3R application pipeline, cu runtime/output/limitări din experimente locale | `helper/script_for_plotting/tabelul_2_1_comparatia_principalelor_solutii_analizate.py` |
| `[FIGURA 3.1 - Selectia cadrelor pe axa temporala]` | Capitolul 3 | Distribuția celor 180 de cadre selectate pentru scena reprezentativă pe axa video originală | `helper/script_for_plotting/figura_3_1_selectia_cadrelor_pe_axa_temporala.py` |
| `[FIGURA 3.2 - Distributia scorurilor de prefiltrare]` | Capitolul 3 | Histograma scorurilor pentru 696 cadre candidate și suprapunerea celor 80 cadre selectate | `helper/script_for_plotting/figura_3_2_distributia_scorurilor_de_prefiltrare.py` |
| `[TABELUL 5.1 - Parametrii principali ai pipeline-ului]` | Capitolul 5 | Parametrii activi din `src/config/settings.yaml`: cadre, selecție, rezoluție, dtype, filtrare, scalare, Gaussian/mesh | `helper/script_for_plotting/tabelul_5_1_parametrii_principali_ai_pipeline_ului.py` |
| `[FIGURA 6.1 - Runtime in functie de numarul de cadre]` | Capitolul 6 | Relația dintre numărul de cadre selectate și timpul total de execuție în run-urile Fast3R dual-output | `helper/script_for_plotting/figura_6_1_runtime_in_functie_de_numarul_de_cadre.py` |
| `[FIGURA 6.2 - Timpii pe etape pentru rulari reprezentative]` | Capitolul 6 | Bar chart stacked cu selecție cadre, inferență, scalare, procesare comună, Gaussian și mesh | `helper/script_for_plotting/figura_6_2_timpii_pe_etape_pentru_rulari_reprezentative.py` |
| `[TABELUL 6.1 - Sumarul scenelor reconstruite local]` | Capitolul 6 | Scenele din `src/outputs`, cu cadre, puncte brute/finale, scală, Gaussians și mesh | `helper/script_for_plotting/tabelul_6_1_sumarul_scenelor_reconstruite_local.py` |
| `[FIGURA 6.3 - Retentia punctelor pentru scena reprezentativa]` | Capitolul 6 | Reducerea punctelor de la predicția densă Fast3R la outputul final procesat pentru `scene_20260422_171717` | `helper/script_for_plotting/figura_6_3_retentia_punctelor_pentru_scena_reprezentativa.py` |
| `[TABELUL 6.2 - Calitatea meshului pentru scena reprezentativa]` | Capitolul 6 | Puncte folosite, triunghiuri, componente, watertight/manifold/self-intersection pentru mesh-ul reprezentativ | `helper/script_for_plotting/tabelul_6_2_calitatea_meshului_pentru_scena_reprezentativa.py` |
| `[FIGURA 6.4 - Curba de validare pentru fine-tuning GA-head]` | Capitolul 6 | Curba `val/loss` pentru run-ul `arkitscenes_10hour_ga_head`, cu primul, ultimul și cel mai bun punct de validare | `helper/script_for_plotting/figura_6_4_curba_de_validare_pentru_fine_tuning_ga_head.py` |
| `[TABELUL 6.3 - Valorile de validare pentru fine-tuning GA-head]` | Capitolul 6 | Valorile numerice de validare și reducerea relativă față de primul punct validat | `helper/script_for_plotting/tabelul_6_3_valorile_de_validare_pentru_fine_tuning_ga_head.py` |

### Registru de figuri manuale care trebuie produse în afara scripturilor

Aceste figuri nu sunt generate automat din CSV/JSON, dar sunt importante pentru ca lucrarea să nu rămână doar o colecție de grafice. Ele trebuie produse manual prin capturi de ecran, diagrame în PowerPoint/draw.io sau randări controlate. Pentru figurile care arată outputul modelului, regula rămâne strictă: se folosesc doar outputuri Fast3R regenerate cu greutățile noi/fine-tuned.

| Marcaj de inserat în teză | Capitol recomandat | Ce trebuie produs manual | Condiție de folosire |
| --- | --- | --- | --- |
| `[FIGURA 1.2 - Fluxul general al lucrarii de la video la scena inspectabila]` | Capitolul 1 | Diagramă conceptuală cu intrare video, selecție cadre, Fast3R, output Gaussian, output mesh și evaluare | Se poate produce imediat ca diagramă manuală |
| `[FIGURA 2.1 - Pozitionarea Fast3R fata de reconstructia iterativa clasica]` | Capitolul 2 | Diagramă comparativă: forward-pass multiview vs COLMAP/MVS/training iterativ | Se poate produce imediat ca diagramă manuală |
| `[FIGURA 5.1 - Arhitectura aplicatiei desktop]` | Capitolul 5 | Diagramă UML/componentă: Presentation/PyQt6, Application Services, Pipeline Layer, Infrastructure/Files | `thesis_docs/document/figures/application/figura_5_1_arhitectura_aplicatiei_desktop.svg` |
| `[FIGURA 5.2 - Fluxul UI scene-first al aplicatiei]` | Capitolul 5 | Diagramă workflow: New Scene, Analyze Video, Run Reconstruction, Open Output, Scene Library, Reconstruction Properties, central viewer | `thesis_docs/document/figures/application/figura_5_2_fluxul_ui_scene_first.svg` |
| `[FIGURA 5.3 - Structura outputului pentru o scena reconstruita]` | Capitolul 5 | Diagramă de structură: metadata, rapoarte, pointcloud, frames, gaussian_splat, mesh optional | `thesis_docs/document/figures/application/figura_5_3_structura_outputului_scene.svg` |
| `[FIGURA 5.4 - Interfata principala a aplicatiei desktop]` | Capitolul 5 | Captură de ecran cu aplicația pornită: Scene Library, workspace central, Reconstruction Properties, tracker de progres | Se produce local din aplicație |
| `[FIGURA 5.5 - Dialogul New Scene si selectia profilului de greutati]` | Capitolul 5 | Captură de ecran cu dialogul New Scene: nume, video, descriere, pipeline, Default/Fine-Tuned Fast3R, preset, taguri | Se produce local din aplicație |
| `[FIGURA 5.6 - Scene Library si panoul Reconstruction Properties]` | Capitolul 5 | Captură focalizată pe managementul scenelor și configurarea reconstrucției | Se produce local din aplicație |
| `[FIGURA 5.7 - Viewerul Viser incorporat pentru scena reconstruita]` | Capitolul 5 | Captură a viewerului Viser încărcat în QWebEngineView; poate arăta doar output regenerat cu greutățile noi dacă scena 3D este vizibilă | Se produce local; respectă regula pentru outputuri vizuale |
| `[FIGURA 6.5 - Montaj cu cadrele selectate pentru scena evaluata]` | Capitolul 6 | Montaj vizual cu 8-12 cadre reprezentative selectate din input | Se poate produce din cadrele selectate; nu este output al modelului |
| `[FIGURA 6.6 - Output Gaussian Fast3R cu greutatile noi]` | Capitolul 6 | Screenshot/randare din viewer pentru scena reconstruită cu checkpoint-ul nou | Se folosește numai după regenerare cu greutățile noi |
| `[FIGURA 6.7 - Output mesh Fast3R cu greutatile noi]` | Capitolul 6 | Screenshot/randare a mesh-ului pentru aceeași scenă și aceleași greutăți noi | Se folosește numai după regenerare cu greutățile noi |
| `[FIGURA 6.8 - Comparatie vizuala intre output Gaussian si output mesh]` | Capitolul 6 | Două panouri alăturate: Gaussian vs mesh pentru aceeași scenă | Se folosește numai după regenerare cu greutățile noi |
| `[FIGURA 6.9 - Exemplu de limitare: memorie sau output partial]` | Capitolul 6 | Captură de log/eroare sau randare parțială care arată limita high-detail/OOM | Dacă este screenshot de output, trebuie regenerat cu greutățile noi; logurile pot fi folosite ca evidență separată |
| `[FIGURA 7.1 - Sinteza limitarilor si directiilor viitoare]` | Capitolul 7 | Diagramă finală cu limitări: scală, memorie, topologie mesh, lipsă benchmark controlat, output visual nou necesar | Se poate produce imediat ca diagramă manuală |

### Marcaje extrase din draftul `experiments/gpt`

Fișierul `experiments/gpt` conține 26 de marcaje de tip `[FIGURA ...]` și `[TABELUL ...]`. Acestea trebuie păstrate ca idei de figuri/tabele pentru textul final. Unele pot fi create din datele proiectului, iar altele sunt figuri conceptuale sau de context, deci trebuie desenate manual ori recreate cu citare corespunzătoare. Numerotarea trebuie reconciliată la final cu registrul de mai sus.

| Marcaj din draft | Tip | Cum se produce în siguranță |
| --- | --- | --- |
| `[FIGURA 1.3 – Contribuțiile principale ale lucrării]` | Conceptual/manual | Poate folosi aceeași idee ca `[FIGURA 1.1 - Contributiile principale ale lucrarii]`; diagramă proprie, nu depinde de date externe |
| `[FIGURA 1.4 – Metodologia generală de dezvoltare a sistemului]` | Conceptual/manual | Diagramă incrementală: analiză soluții, alegere Fast3R, selecție cadre, postprocesare, scalare, outputuri, fine-tuning |
| `[FIGURA 2.1 – Fluxul general al tehnologiilor utilizate în cadrul sistemului]` | Conceptual/manual | Diagramă proprie cu video, selecție, Fast3R, Open3D, Gaussian, mesh, PyQt6 |
| `[FIGURA 2.2 – Evoluția abordărilor de reconstrucție 3D]` | Context extern/manual | Timeline recreat de noi: SfM/MVS, depth monocular, DUSt3R/Fast3R, Gaussian Splatting; necesită citări, nu copiere de figuri din lucrări |
| `[FIGURA 2.3 – Ramificarea pipeline-ului către reprezentările Gaussian și Mesh]` | Conceptual/manual | Diagramă proprie despre ramificarea după reconstrucția comună |
| `[FIGURA 2.4 – Exemple de cadre cu niveluri diferite de calitate]` | Manual/local | Colaj din cadre proprii: clar, blur, supraexpus, textură slabă |
| `[FIGURA 2.5 – Procesul de selecție geometry-aware]` | Conceptual/manual | Diagramă proprie cu scor vizual, deduplicare, probă geometrică, selecție finală |
| `[FIGURA 2.6 – Exemplu de nor de puncte generat de sistem]` | Output model | Se folosește numai dacă este regenerat cu greutățile noi Fast3R |
| `[FIGURA 2.7 – Efectul normalizării scalei]` | Manual/local | Poate fi diagramă/plot din metadate; dacă arată output 3D vizual, trebuie să fie new-weight |
| `[FIGURA 2.8 – Exemplu de reprezentare Gaussian Splatting]` | Output model | Se folosește numai pentru Gaussian generat cu greutățile noi |
| `[FIGURA 2.9 – Generarea unui model Mesh prin reconstrucție Poisson]` | Conceptual/manual | Diagramă proprie sau schemă Open3D/Poisson cu citare; nu necesită screenshot de output |
| `[FIGURA 3.1 – Fluxul datelor în cadrul sistemului]` | Conceptual/manual | Diagramă proprie despre traseul datelor de la video la output |
| `[FIGURA 3.2 – Exemple de cadre eliminate în etapa de preprocesare]` | Manual/local | Colaj din cadre proprii eliminate pentru blur, expunere, lipsă de textură |
| `[FIGURA 3.3 – Procesul de selecție geometry-aware]` | Conceptual/manual | Poate fi combinată cu `[FIGURA 2.5 – Procesul de selecție geometry-aware]` pentru a evita duplicarea |
| `[FIGURA 3.4 – Efectele filtrării și normalizării]` | Output/model sau metadate | Dacă este vizual 3D, numai new-weight; alternativ poate fi diagramă/metadate fără captură de output |
| `[FIGURA 3.5 – Fluxul de pregătire a datelor pentru fine-tuning]` | Conceptual/manual | Diagramă proprie despre ARKitScenes, metadate, split, training/validation |
| `[FIGURA 4.1 – Poziția modelului Fast3R în cadrul pipeline-ului]` | Conceptual/manual | Diagramă proprie despre rolul Fast3R între selecția cadrelor și postprocesare |
| `[FIGURA 4.2 – Fluxul de integrare al modelului Fast3R]` | Conceptual/manual | Diagramă proprie despre apelul modelului, inferență, filtrare, outputuri |
| `[FIGURA 4.3 – Strategia de adaptare a modelului]` | Conceptual/manual | Diagramă head-only fine-tuning: backbone înghețat, head antrenat, ARKitScenes |
| `[TABELUL 2.2 – Justificarea tehnologiilor utilizate]` | Manual/local | Tabel de argumentare: Fast3R, Open3D, PyTorch, PyQt6, Gaussian, mesh, YAML/JSON |
| `[TABELUL 2.3 – Impactul strategiilor de selecție a cadrelor]` | Date locale, dacă există comparații | Se folosește doar dacă sunt disponibile rulari comparabile; altfel se transformă în discuție calitativă |
| `[TABELUL 2.4 – Comparația dintre reprezentările Point Cloud, Gaussian Splatting și Mesh]` | Conceptual/manual | Tabel teoretic cu avantaje, limitări și utilizări |
| `[TABELUL 3.1 – Statistici ale procesului de selecție]` | Date locale | Se poate deriva din `scene_metadata.json` și prefilter CSV-uri |
| `[TABELUL 4.1 – Parametrii principali utilizați în procesul de inferență]` | Date locale | Se poate alinia cu `[TABELUL 5.1 - Parametrii principali ai pipeline-ului]` |
| `[TABELUL 4.2 – Configurația utilizată pentru fine-tuning]` | Date locale | Tabel din configurația Vast/ARKitScenes/GA-head; trebuie păstrat ca extensie, nu contribuție centrală |

## 11. Bibliografie validabilă local

Sursele pentru care există urme suficient de clare în repository sunt:

1. Jianing Yang et al., *Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass*, CVPR 2025.
2. Michael Kazhdan, Matthew Bolitho, Hugues Hoppe, *Poisson Surface Reconstruction*, Symposium on Geometry Processing, 2006.
3. Michael Kazhdan, Hugues Hoppe, *Screened Poisson Surface Reconstruction*, ACM Transactions on Graphics / SIGGRAPH, 2013.
4. Documentația Open3D pentru reconstrucție de suprafețe.
5. Documentația oficială ARKitScenes și materialele de preprocesare folosite de proiectul Fast3R.

Orice alte intrări bibliografice suplimentare trebuie verificate și completate înainte de forma finală de predare.

## 12. Revizuiri manuale necesare înainte de predare

- confirmarea titlului final cu formulare academică preferată de coordonator;
- completarea bibliografiei într-un format unitar cerut de facultate;
- inserarea efectivă a figurilor și capturilor de ecran;
- confirmarea dacă se dorește păstrarea subcapitolului de fine-tuning la dimensiune mare sau într-o formă mai scurtă;
- verificarea ultimei comparații „curate” de 4000 de pași, dacă apar artefactele finale complete și se dorește introducerea ei explicită.
