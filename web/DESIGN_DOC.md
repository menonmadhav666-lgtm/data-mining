# 🗞️ The Daily Classifier — Vintage Broadsheet Web Application
## Design & Technical Architecture Document

---

### 1. Executive Summary & Vision
**The Daily Classifier** is an editorial-grade, vintage broadsheet newspaper web interface designed for real-time news text classification powered by a TF-IDF + Multinomial Naive Bayes machine learning pipeline.

The design evokes the golden age of print journalism (late 19th / early 20th-century broadsheets, telegraphic dispatches, and letterpress typography). Modern machine learning telemetry—such as category probabilities, vectorizer n-grams, and evaluation metrics—is reimagined through antique editorial metaphors: printing presses, ink stamps, classified ads, ticker tapes, and woodblock engravings.

---

### 2. Aesthetic & Visual Direction

#### 2.1 Design Philosophy: "The Living Broadsheet"
- **Aged Paper & Tactile Textures**: Textured newsprint backgrounds with subtle paper grain, aged patina, faint ink bleed, and deckled border rules.
- **Letterpress & Print Artifacts**: High-contrast black printer's ink on parchment, woodcut decorative dividers, drop caps, ornate corner brackets, and red wax / ink rubber stamps for classification verdict tags.
- **Multi-Column Editorial Grid**: Strict 3-to-4 column newspaper grid layout with vertical rule dividers, justified columns, and dynamic headline scaling.

#### 2.2 Color Palette

| Token | Hex Value | Name | Role & Usage |
| :--- | :--- | :--- | :--- |
| `--bg-parchment` | `#F5EFE0` | Antique Newsprint | Primary page background |
| `--bg-parchment-dark`| `#E8DEC8` | Aged Paper Shadow | Card surfaces, ledger insets, borders |
| `--ink-primary` | `#1F1A17` | Printer's Lampblack | Headlines, masthead, high-priority text |
| `--ink-secondary` | `#4A3E35` | Walnut Sepia | Body copy, subheadings, column rules |
| `--ink-faded` | `#7D6F63` | Weathered Ink | Timestamps, footnotes, issue numbers |
| `--stamp-red` | `#96281B` | Editorial Crimson Stamp | Breaking News, Politics, Urgent Badges |
| `--stamp-green` | `#2B5838` | Press Forest Stamp | Sport, High Confidence Seals |
| `--stamp-blue` | `#224870` | Telegraphic Indigo | Tech, System Telemetry |
| `--stamp-amber` | `#A05A1C` | Brass Amber Stamp | Business, Market Indices |
| `--stamp-purple` | `#5C2D63` | Royal Broadsheet | Entertainment & Arts |
| `--rule-color` | `#2E2620` | Engraved Rule Line | Column split lines, banner borders |

---

### 3. Typography Hierarchy & Masthead Styling

The application leverages classic Google Web Fonts paired to simulate an authentic linotype-printed newspaper.

```
┌────────────────────────────────────────────────────────────────────────┐
│   THE DAILY CLASSIFIER  (Masthead: UnifrakturMaguntia / Pirata One)    │
│   "All The News That Is Fit To Vectorize" • Est. 1922 • Vol. CIV No. 42│
├────────────────────────────────────────────────────────────────────────┤
│   HEADLINE (Playfair Display SC / Old Standard TT - 36-48px Bold)      │
│   Subhead / Kicker (Cinzel / Cormorant Garamond - 18px Italic Small Caps│
│   Body Text (Libre Baskerville / IM Fell English - 14-16px Justified)   │
│   Telemetry & Ledgers (Courier Prime / Special Elite - Typewriter)     │
└────────────────────────────────────────────────────────────────────────┘
```

#### Font Stack:
1. **Masthead Title & Letterhead:** `'Playfair Display'`, `'Pirata One'`, or `'UnifrakturMaguntia'` with ornate ornamental flourishes.
2. **Main Headlines & Section Kickers:** `'Playfair Display'`, `'Cinzel'`, and `'Old Standard TT'`.
3. **Article Copy & Editorial Text:** `'Libre Baskerville'`, `'Cormorant Garamond'`, or `'Merriweather'` with CSS `text-align: justify` and `text-justify: inter-word`.
4. **Interactive Inputs, Telegrams & Dispatches:** `'Special Elite'` or `'Courier Prime'` to replicate mechanical typewriter keystrokes.

---

### 4. Page Architecture & Section Layout

```
+========================================================================+
|                       [TOP DATE & MOTTO BANNER]                        |
| Weather: Foggy, 54°F • London & Worldwide Edition • Price: 2 Pence    |
+========================================================================+
|                                                                        |
|                     𝕿𝖍𝖊 𝕯𝖆𝖎𝖑𝖞 𝕮𝖑𝖆𝖘𝖘𝖎𝖋𝖎𝖊𝖗                       |
|           VOICE OF THE INTELLIGENT PRESS • AUTOMATED BY NAIVE BAYES    |
|                                                                        |
+========================================================================+
| [TICKER TAPE]: BREAKING NEWS • MODEL ACCURACY AT 97.4% • 5 TOPICS READ |
+========================================================================+
|                                  |                                     |
|  COLUMN 1: THE TYPESETTER'S DESK |  COLUMN 2: THE DISPATCH VERDICT     |
|  (Interactive Input Area)        |  (Classification & Stamp Result)    |
|                                  |                                     |
|  • Manuscript Textarea           |  • [RUBBER STAMP BADGE: "POLITICS"] |
|  • Sample Dispatch Telegrams     |  • Confidence Gauge (Antique Meter) |
|    - Sport Derby Dispatch        |  • Probability Distribution Ledger  |
|    - Tech Silicon Breakthrough   |    (Business, Tech, Sport, etc.)    |
|    - Business Market Rally       |  • Top Keyword Signals (TF-IDF Sal.)|
|  • "Engage The Rotary Press" btn |                                     |
|                                  |                                     |
+----------------------------------+-------------------------------------+
|                                                                        |
|  COLUMN 3 (SPANNING): THE EDITORIAL AUDIT & ARCHIVAL METRICS           |
|  • Model Metrics: Accuracy (97.4%), Precision (0.97), Recall (0.97)    |
|  • Confusion Matrix & Confusion Ledger (Classified Distribution)       |
|  • Feature Lexicon Highlights (Top n-grams per department)             |
|                                                                        |
+========================================================================+
|  FOOTER: "Typeset with Care • Antigravity Press • Archive No. 2026"     |
+========================================================================+
```

---

### 5. Detailed Component Specifications

#### 5.1 The Historic Letterhead & Masthead
- **Ornate Triple Rule**: Thick center line flanked by thin hair lines with antique diamond rosettes.
- **Date & Edition Bar**: Dynamic real-time date formatted as `"Friday, the 9th of October, Anno Domini 2026"`.
- **Weather & Market Widget**: `"Barometer: 30.12 in • Press Condition: Prime • Model State: Tuned & Ready"`.

#### 5.2 The Typesetter's Slate (Input Console)
- Styled as a vintage typewriter paper feed with ruled lines and a subtle cursor blink.
- Pre-loaded "Telegram Snippets" (quick sample buttons styled as Western Union telegram cards).
- Clear, Paste, and Classify buttons styled with embossed beveled brass/woodblock edges.

#### 5.3 The Telegraphic Verdict (Classification Display)
- **Rubber Stamp Animation**: Stamped verdict slamming down at a slight random angle (`rotate(-4deg)`) with distressed ink edge texture.
- **Category Badge**: Distinct vintage woodcut borders tailored to each department:
  - 🏛️ *Politics* (Crimson)
  - ⚽ *Sport* (Forest Ink)
  - 💼 *Business* (Amber)
  - ⚡ *Tech* (Indigo)
  - 🎭 *Entertainment* (Plum)
- **Galvanometer / Probability Ledger**: Antique horizontal bar gauges showing probability percentages from 0% to 100%.

#### 5.4 The Editorial Audit (Model Performance Section)
- Tabular display resembling a 19th-century stock and trade ledger.
- Displays TF-IDF vocabulary size (over 14,000 n-grams analyzed), test accuracy, F1-scores, and training parameters (`alpha=0.1`, `ngram_range=(1,2)`).

---

### 6. Technical Stack & Implementation Strategy

```
web/
├── index.html          # Semantic HTML5 broadsheet structure with microdata
├── styles.css          # Vintage typography, paper textures, letterpress CSS
├── app.js              # Client-side UI logic, samples, and API integration
├── api.py              # Lightweight Flask/FastAPI backend serving Naive Bayes
└── assets/             # Textures (paper grain SVG), woodcut ornaments, SVG stamps
```

#### Key Technical Highlights:
1. **Zero External Heavy Framework Overhead**: High-speed Vanilla HTML5 + CSS Grid + Modern JS for optimal rendering and smooth 60fps stamping animations.
2. **Self-Contained Mode & Live Backend Mode**:
   - *Standalone Mode*: Pre-bundled client model weights or fallback heuristic simulator for instantaneous static preview.
   - *Live Backend Mode*: Connects to `classify.py` / `api.py` endpoint for exact scikit-learn pipeline inference.
3. **Print & Export Feature**: `@media print` CSS sheet formatting the current classification into an authentic printable 1-page vintage morning paper.

---

### 7. Next Steps & Development Roadmap
- [x] **Phase 1**: Architecture & Visual Design Document (`web/DESIGN_DOC.md`).
- [ ] **Phase 2**: Core HTML structure (`web/index.html`) & Masthead typography.
- [ ] **Phase 3**: Broadsheet layout and vintage ink CSS styling (`web/styles.css`).
- [ ] **Phase 4**: Interactive client script (`web/app.js`) with sample telegrams & probability animations.
- [ ] **Phase 5**: Python web server / API integration (`web/server.py`) connecting the scikit-learn model.
