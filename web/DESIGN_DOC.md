# 🗞️ The Daily Classifier — Vintage Broadsheet Web Application
## Design & Technical Architecture Document

---

### 1. Executive Summary & Vision
**The Daily Classifier** is an editorial-grade, vintage broadsheet newspaper web interface designed for real-time news text classification powered by a TF-IDF + Multinomial Naive Bayes machine learning pipeline.

The application combines the tactile nostalgia of early 20th-century print journalism with a **modular, drag-and-drop broadsheet grid system**. Every newspaper section (the "Typesetter's Slate", "Telegraphic Verdict", "Probability Ledger", "Editorial Audit", "Classifieds Lexicon", etc.) is an interactive **draggable tile**. Users and editors can rearrange the broadsheet layout dynamically on the fly, and click any tile to open an in-depth **Editorial Inspector / Edit View** to customize its contents, parameters, column span, and typography.

---

### 2. Aesthetic & Visual Direction

#### 2.1 Design Philosophy: "The Living Broadsheet"
- **Aged Paper & Tactile Textures**: Textured newsprint backgrounds with subtle paper grain, aged patina, faint ink bleed, and deckled border rules.
- **Letterpress & Print Artifacts**: High-contrast black printer's ink on parchment, woodcut decorative dividers, drop caps, ornate corner brackets, and red wax / ink rubber stamps for classification verdict tags.
- **Modular Draggable Grid**: A flexible multi-column broadsheet layout with ink dividers where every article box/tile can be picked up, re-ordered with smooth dragging physics, and saved.

#### 2.2 Color Palette & Design Tokens

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
| `--drag-ghost` | `rgba(150,40,27,0.15)` | Linotype Layout Ghost | Drag-and-drop placeholder indicator |

---

### 3. Modular Tile & Drag-and-Drop Grid System

The broadsheet is constructed from modular **Editorial Tiles** positioned within an interactive 12-column responsive print grid.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TOP MASTHEAD & LETTERHEAD                       │
├────────────────────────────────────────────────────────────────────────┤
│ [DRAG-AND-DROP BROADSHEET GRID WORKSPACE]                             │
│                                                                        │
│ ┌──────────────────────────┐ ┌───────────────────────────────────────┐ │
│ │ ⠿ TILE: TYPESETTER SLATE  │ │ ⠿ TILE: TELEGRAPHIC VERDICT           │ │
│ │ (Input & Telegrams)      │ │ (Rubber Stamp & Probabilities)        │ │
│ │ [Col Span: 6] [Click-Edit]│ │ [Col Span: 6] [Click-Edit]            │ │
│ └──────────────────────────┘ └───────────────────────────────────────┘ │
│                                                                        │
│ ┌──────────────────────────┐ ┌───────────────────────────────────────┐ │
│ │ ⠿ TILE: MODEL AUDIT      │ │ ⠿ TILE: FEATURE LEXICON & N-GRAMS     │ │
│ │ (Test Accuracy 97.4%)    │ │ (Salient Words per Topic)             │ │
│ │ [Col Span: 7] [Click-Edit]│ │ [Col Span: 5] [Click-Edit]            │ │
│ └──────────────────────────┘ └───────────────────────────────────────┘ │
│                                                                        │
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │ ⠿ TILE: CONFUSION MATRIX & HISTORICAL ARCHIVES (Col Span: 12)       │ │
│ └────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

#### 3.1 Drag-and-Drop Interaction Mechanics
1. **Grab Handles & Hover States**: Each tile features an authentic printer's ornament grab handle (`⠿` linotype mark). Hovering reveals an engraved highlight border.
2. **Re-ordering & Drop Indicators**: Dragging a tile provides visual ghosting with a dashed printer's registration line indicating the new drop slot.
3. **Layout Persistence**: Custom layout order and column widths are automatically saved to `localStorage` (and synchronized with the backend configuration).
4. **Layout Presets**:
   - *Default Morning Edition* (Standard 2-column lead + bottom telemetry).
   - *Analyst's Ledger* (Metrics & probability matrices front and center).
   - *Minimalist Dispatch* (Compact input slate + immediate rubber stamp verdict).

---

### 4. Click-to-Edit Modal / Tile Inspector Page

Clicking on any tile header or edit icon opens the **"Typesetter's Linotype Inspector"** modal / edit view:

```
+========================================================================+
| 🪶 THE TYPESETTER'S DESK — TILE CONFIGURATION & EDIT VIEW             |
+========================================================================+
| Tile Name:       [ Telegraphic Dispatch Verdict                       ] |
| Column Span:     ( ) 4 Cols  (•) 6 Cols  ( ) 8 Cols  ( ) 12 Cols (Full) |
| Display Header:  [ BREAKING DISPATCH & CLASSIFIED TELEGRAPH           ] |
| Subhead Kicker:  [ Transmitted via Naive Bayes Rotary Telegraph       ] |
| Section Theme:   [ Crimson Broadsheet / Gold Foil / Ink Sepia    ▼ ]    |
| Custom Content:  [ Enable Probability Gauges [✓] Enable Audio [✓]     ] |
| Data Filter:     [ All Categories (Sport, Tech, Business, Pol, Ent)  ▼] |
+------------------------------------------------------------------------+
| [ Save & Typeset Tile ]    [ Reset to Default ]    [ Dismiss Inspector ]|
+========================================================================+
```

#### Editable Attributes per Tile:
- **Title & Kicker Text**: Customize the vintage headlines and subheadings.
- **Column Span & Width**: Change grid width (`span 4`, `span 6`, `span 8`, `span 12`).
- **Telemetry Display Toggles**: Toggle confidence percentage, bar gauges, raw logits, or timestamp badges.
- **Custom Notes / Editorial Commentary**: Inject custom editorial copy directly into the tile.
- **Visual Styling**: Choose vintage ink accent borders (Crimson, Forest Ink, Indigo, Sepia).

---

### 5. Standard Tile Catalog

| Tile ID | Title | Default Span | Primary Role |
| :--- | :--- | :--- | :--- |
| `tile-input` | **The Typesetter's Slate** | 6 Columns | Interactive typewriter input + sample telegrams |
| `tile-verdict` | **The Telegraphic Verdict** | 6 Columns | Stamped category badge + probability galvanometer |
| `tile-audit` | **The Editorial Audit** | 7 Columns | Accuracy (97.4%), Precision, Recall & F1 tables |
| `tile-lexicon` | **Classifieds & Top N-Grams** | 5 Columns | Top keywords driving classification decisions |
| `tile-matrix` | **Confusion Matrix Gazette** | 12 Columns | Visual matrix of predicted vs actual test samples |
| `tile-ticker` | **Telegraphic News Wire** | 12 Columns | Real-time scrolling vintage stock/news ticker |

---

### 6. Technical Stack & Implementation Plan

```
web/
├── DESIGN_DOC.md       # Architecture specification with drag-and-drop & edit specs
├── index.html          # Semantic broadsheet DOM with draggable tile containers
├── styles.css          # Newspaper styling, drag physics, linotype inspector modal
├── app.js              # Drag-and-drop engine, tile inspector modal, ML predictions
├── server.py           # Python HTTP / REST API connecting scikit-learn pipeline
└── assets/             # Textures & vintage woodcut assets
```

---

### 7. Phased Implementation Roadmap
- [x] **Phase 1**: Updated Design Document with Draggable Tiles & Edit Modal (`web/DESIGN_DOC.md`).
- [ ] **Phase 2**: HTML Skeleton with Draggable Tile Components (`web/index.html`).
- [ ] **Phase 3**: Vintage Broadsheet Styles, Drag Shadows & Linotype Modal CSS (`web/styles.css`).
- [ ] **Phase 4**: Drag-and-Drop Engine, State Management & Tile Editor JS (`web/app.js`).
- [ ] **Phase 5**: Integrated Python Model Server (`web/server.py`).
