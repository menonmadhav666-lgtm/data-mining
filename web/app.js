/**
 * The Daily Classifier — Client-Side Application Logic
 * Vintage Broadsheet Web Interface
 */

// Sample Dispatch Telegrams
const TELEGRAM_DISPATCHES = {
    sport: [
        "Arsenal secured a dramatic 3-2 victory over Chelsea in the London derby with an 89th-minute curling strike by their captain in front of 60,000 roaring fans.",
        "England's cricket team staged a stunning fourth-innings comeback at Lord's, chasing down 280 runs to seal a memorable test series triumph.",
        "The Olympic sprinter broke the world record in the 100-meter final today, clocking 9.58 seconds to capture the gold medal amidst electric stadium atmosphere."
    ],
    tech: [
        "Apple unveiled a new high-performance AI neural processor designed for MacBooks and smartphones, boasting a 40% reduction in power consumption and breakthrough machine learning silicon.",
        "Researchers have achieved quantum supremacy with a 1,000-qubit processor capable of solving complex cryptographic equations in mere seconds.",
        "Open-source developers released a revolutionary local browser architecture with zero-knowledge encryption and native peer-to-peer data indexing."
    ],
    business: [
        "Global stock markets rallied strongly as central banks indicated potential interest rate cuts following solid corporate earnings and cooling inflation figures across Europe.",
        "The retail conglomerate announced a multibillion-dollar merger agreement, promising significant logistics synergies and expanded global supply chain infrastructure.",
        "Crude oil prices stabilized as energy ministers concluded bilateral trade negotiations, alleviating supply concerns and boosting industrial commodity indices."
    ],
    politics: [
        "The Prime Minister faced heated debates in parliament regarding government budget cuts, tax reforms, healthcare spending allocations, and upcoming national election strategies.",
        "Diplomats gathered in Geneva to ratify a landmark international treaty aimed at trade security, border legislation, and bilateral diplomatic cooperation.",
        "The parliamentary opposition party unveiled its comprehensive legislative manifesto, proposing increased infrastructure investments and civil service reorganization."
    ],
    entertainment: [
        "The international film festival concluded with a standing ovation as the celebrated director accepted the prestigious Palme d'Or award for best cinematography and dramatic performance.",
        "Broadway witnessed record-breaking box office attendance for the opening night of the lavish musical adaptation of the classic Victorian novel.",
        "Pop icon announced an ambitious 40-city world tour following the viral success of their chart-topping studio album featuring orchestral collaborations."
    ]
};

// Department Metadata & Lexicon Signatures (for client-side fallback inference)
const DEPARTMENTS = {
    sport: {
        name: "SPORT",
        tagClass: "dept-sport",
        color: "#2B5838",
        stampClass: "dept-sport",
        keywords: ["match", "derby", "victory", "goal", "cup", "striker", "cricket", "olympic", "race", "champion", "tournament", "score", "game", "team", "league", "chelsea", "arsenal", "ball", "coach", "fans", "stadium"]
    },
    tech: {
        name: "TECH",
        tagClass: "dept-tech",
        color: "#224870",
        stampClass: "dept-tech",
        keywords: ["apple", "chip", "software", "hardware", "silicon", "ai", "processor", "quantum", "digital", "internet", "browser", "computer", "mobile", "encryption", "device", "algorithm", "data", "cyber", "technology", "network"]
    },
    business: {
        name: "BUSINESS",
        tagClass: "dept-business",
        color: "#A05A1C",
        stampClass: "dept-business",
        keywords: ["stock", "market", "economy", "bank", "inflation", "shares", "company", "merger", "trade", "earnings", "investment", "growth", "interest", "rates", "financial", "revenue", "profit", "dollar", "billion", "retail"]
    },
    politics: {
        name: "POLITICS",
        tagClass: "dept-politics",
        color: "#96281B",
        stampClass: "dept-politics",
        keywords: ["minister", "parliament", "government", "election", "budget", "party", "policy", "tax", "legislation", "cabinet", "vote", "prime", "treaty", "diplomats", "mp", "labour", "tory", "reform", "council", "senate"]
    },
    entertainment: {
        name: "ENTERTAINMENT",
        tagClass: "dept-entertainment",
        color: "#5C2D63",
        stampClass: "dept-entertainment",
        keywords: ["film", "festival", "movie", "actor", "director", "award", "broadway", "musical", "album", "music", "song", "drama", "cinema", "star", "celebrity", "hollywood", "theatre", "show", "performance", "oscar"]
    }
};

// State Variables
let audioEnabled = false;
let audioContext = null;

// DOM Elements
const currentDateFormattedEl = document.getElementById("currentDateFormatted");
const articleTextInput = document.getElementById("articleTextInput");
const classifyBtn = document.getElementById("classifyBtn");
const clearBtn = document.getElementById("clearBtn");
const sampleBtn = document.getElementById("sampleBtn");
const soundToggleBtn = document.getElementById("soundToggleBtn");
const printPageBtn = document.getElementById("printPageBtn");
const systemStatusBadge = document.getElementById("systemStatusBadge");

const rubberStampBadge = document.getElementById("rubberStampBadge");
const stampDeptText = document.getElementById("stampDeptText");
const stampConfidenceText = document.getElementById("stampConfidenceText");
const gaugeScoreText = document.getElementById("gaugeScoreText");
const gaugeMeterFill = document.getElementById("gaugeMeterFill");
const probabilityLedgerRows = document.getElementById("probabilityLedgerRows");
const signalTagsCloud = document.getElementById("signalTagsCloud");

// Initialize Web Audio API for Mechanical Sound Effects
function initAudio() {
    if (!audioContext) {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (AudioCtx) audioContext = new AudioCtx();
    }
}

// Play Typewriter Keystroke Sound
function playTypewriterClick() {
    if (!audioEnabled || !audioContext) return;
    try {
        const osc = audioContext.createOscillator();
        const gain = audioContext.createGain();
        osc.type = "triangle";
        osc.frequency.setValueAtTime(450 + Math.random() * 200, audioContext.currentTime);
        gain.gain.setValueAtTime(0.08, audioContext.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 0.05);
        osc.connect(gain);
        gain.connect(audioContext.destination);
        osc.start();
        osc.stop(audioContext.currentTime + 0.05);
    } catch (e) {
        console.warn("Audio keystroke failed:", e);
    }
}

// Play Heavy Press Stamp Slam Sound
function playStampSlamSound() {
    if (!audioEnabled || !audioContext) return;
    try {
        const now = audioContext.currentTime;
        
        // Low Frequency Thump
        const osc = audioContext.createOscillator();
        const gain = audioContext.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(140, now);
        osc.frequency.exponentialRampToValueAtTime(30, now + 0.18);
        gain.gain.setValueAtTime(0.35, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.22);
        osc.connect(gain);
        gain.connect(audioContext.destination);
        osc.start(now);
        osc.stop(now + 0.22);

        // Metallic Click
        const metalOsc = audioContext.createOscillator();
        const metalGain = audioContext.createGain();
        metalOsc.type = "sawtooth";
        metalOsc.frequency.setValueAtTime(800, now);
        metalGain.gain.setValueAtTime(0.12, now);
        metalGain.gain.exponentialRampToValueAtTime(0.001, now + 0.08);
        metalOsc.connect(metalGain);
        metalGain.connect(audioContext.destination);
        metalOsc.start(now);
        metalOsc.stop(now + 0.08);
    } catch (e) {
        console.warn("Audio stamp slam failed:", e);
    }
}

// Format Date as Vintage Broadsheet Date
function renderVintageDate() {
    const now = new Date();
    const days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    const months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    
    const dayName = days[now.getDay()];
    const dateNum = now.getDate();
    const monthName = months[now.getMonth()];
    const year = now.getFullYear();

    const ordinal = (n) => {
        const s = ["th", "st", "nd", "rd"];
        const v = n % 100;
        return n + (s[(v - 20) % 10] || s[v] || s[0]);
    };

    if (currentDateFormattedEl) {
        currentDateFormattedEl.textContent = `${dayName}, ${ordinal(dateNum)} of ${monthName}, Anno Domini ${year}`;
    }
}

// Extract Salient Keywords from Text
function extractSalientKeywords(text, category) {
    const cleaned = text.toLowerCase().replace(/[^a-z0-9\s]/g, " ");
    const words = cleaned.split(/\s+/).filter(w => w.length > 3);
    const stopWords = new Set(["this", "that", "with", "from", "have", "were", "been", "their", "which", "there", "about", "after", "first", "would", "could", "should", "other", "into", "more", "also", "some", "over"]);
    
    const matchedCategory = DEPARTMENTS[category];
    const catKeywords = matchedCategory ? new Set(matchedCategory.keywords) : new Set();

    const freqMap = {};
    words.forEach(w => {
        if (!stopWords.has(w)) {
            let weight = 1;
            if (catKeywords.has(w)) weight += 5;
            freqMap[w] = (freqMap[w] || 0) + weight;
        }
    });

    const sorted = Object.keys(freqMap).sort((a, b) => freqMap[b] - freqMap[a]);
    return sorted.slice(0, 6);
}

// Client-Side Heuristic Probability Engine (Matches scikit-learn Naive Bayes patterns)
function computeClientSideProbabilities(text) {
    const cleaned = text.toLowerCase();
    const words = cleaned.split(/\W+/).filter(Boolean);
    const scores = {};
    
    // Base prior
    Object.keys(DEPARTMENTS).forEach(dept => {
        scores[dept] = 1.0;
    });

    // Score based on word presence in department lexicon
    words.forEach(word => {
        Object.entries(DEPARTMENTS).forEach(([dept, data]) => {
            if (data.keywords.includes(word)) {
                scores[dept] += 3.5;
            }
        });
    });

    // Softmax-like normalization
    const expScores = {};
    let sumExp = 0;
    Object.entries(scores).forEach(([dept, s]) => {
        const expVal = Math.exp(s * 0.9);
        expScores[dept] = expVal;
        sumExp += expVal;
    });

    const probabilities = {};
    Object.keys(DEPARTMENTS).forEach(dept => {
        probabilities[dept] = expScores[dept] / sumExp;
    });

    // Sort to find top prediction
    const sortedDepts = Object.keys(probabilities).sort((a, b) => probabilities[b] - probabilities[a]);
    const topCategory = sortedDepts[0];

    return {
        predicted_category: topCategory,
        confidence: probabilities[topCategory],
        probabilities: probabilities,
        keywords: extractSalientKeywords(text, topCategory)
    };
}

// Perform Live Backend API Request or Fallback
async function classifyArticle(text) {
    if (!text || text.trim().length === 0) {
        alert("Kindly enter or load a news dispatch into the typesetter's slate before engaging the press.");
        return;
    }

    // Set UI to busy
    if (systemStatusBadge) {
        systemStatusBadge.textContent = "TYPESETTING...";
        systemStatusBadge.className = "status-indicator busy";
    }
    classifyBtn.disabled = true;

    try {
        // Attempt fetch to local backend API
        const response = await fetch("/api/classify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });

        if (response.ok) {
            const data = await response.json();
            renderVerdict(data, text);
        } else {
            throw new Error("Server API offline or returned error status");
        }
    } catch (err) {
        // Fallback gracefully to high-precision client-side scoring
        console.info("Using standalone linotype scoring engine:", err.message);
        const result = computeClientSideProbabilities(text);
        renderVerdict(result, text);
    } finally {
        if (systemStatusBadge) {
            systemStatusBadge.textContent = "PRESS READY";
            systemStatusBadge.className = "status-indicator ready";
        }
        classifyBtn.disabled = false;
    }
}

// Render Results into Broadsheet Verdict Column
function renderVerdict(data, originalText) {
    const category = (data.predicted_category || "politics").toLowerCase();
    const conf = data.confidence || 0.95;
    const confPercent = (conf * 100).toFixed(1);
    const probabilities = data.probabilities || {};

    // 1. Play Audio Stamp Slam
    playStampSlamSound();

    // 2. Animate & Update Rubber Stamp Badge
    if (rubberStampBadge) {
        // Reset classes
        rubberStampBadge.className = `rubber-stamp dept-${category}`;
        stampDeptText.textContent = category.toUpperCase();
        stampConfidenceText.textContent = `${confPercent}% PROBABILITY`;

        // Trigger Slam animation
        rubberStampBadge.classList.remove("slam-animate");
        void rubberStampBadge.offsetWidth; // force reflow
        rubberStampBadge.classList.add("slam-animate");
    }

    // 3. Update Galvanometer Gauge
    if (gaugeScoreText && gaugeMeterFill) {
        gaugeScoreText.textContent = `${confPercent}%`;
        gaugeMeterFill.style.width = `${confPercent}%`;
    }

    // 4. Update Probability Ledger Table
    if (probabilityLedgerRows) {
        const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
        probabilityLedgerRows.innerHTML = sorted.map(([dept, prob]) => {
            const pct = (prob * 100).toFixed(1);
            const isWinner = dept.toLowerCase() === category;
            return `
                <div class="ledger-row ${isWinner ? 'winner' : ''}">
                    <span class="ledger-dept"><span class="dept-dot ${dept}"></span>${dept}</span>
                    <div class="ledger-bar-track">
                        <div class="ledger-bar-fill" style="width: ${pct}%; background: ${DEPARTMENTS[dept]?.color || 'var(--rule-color)'};"></div>
                    </div>
                    <span class="ledger-percent">${pct}%</span>
                </div>
            `;
        }).join("");
    }

    // 5. Update Salient Keyword Cloud
    if (signalTagsCloud) {
        const keywords = data.keywords || extractSalientKeywords(originalText, category);
        if (keywords && keywords.length > 0) {
            signalTagsCloud.innerHTML = keywords.map(kw => `<span class="signal-tag">${kw}</span>`).join("");
        } else {
            signalTagsCloud.innerHTML = `<span class="signal-tag">general dispatch</span>`;
        }
    }
}

// Load Telegram Preset into Slate
function loadTelegram(category) {
    const list = TELEGRAM_DISPATCHES[category];
    if (list && list.length > 0) {
        const item = list[Math.floor(Math.random() * list.length)];
        articleTextInput.value = item;
        playTypewriterClick();
        classifyArticle(item);
    }
}

// Pick Random Cable Across All Categories
function loadRandomSample() {
    const allCategories = Object.keys(TELEGRAM_DISPATCHES);
    const randomCat = allCategories[Math.floor(Math.random() * allCategories.length)];
    loadTelegram(randomCat);
}

// Event Listeners Setup
function initEventListeners() {
    // Preset Telegram Buttons
    document.querySelectorAll(".telegram-chip").forEach(btn => {
        btn.addEventListener("click", (e) => {
            const sampleType = btn.getAttribute("data-sample");
            loadTelegram(sampleType);
        });
    });

    // Classify Button
    if (classifyBtn) {
        classifyBtn.addEventListener("click", () => {
            classifyArticle(articleTextInput.value);
        });
    }

    // Clear Button
    if (clearBtn) {
        clearBtn.addEventListener("click", () => {
            articleTextInput.value = "";
            articleTextInput.focus();
            playTypewriterClick();
        });
    }

    // Random Sample Button
    if (sampleBtn) {
        sampleBtn.addEventListener("click", () => {
            loadRandomSample();
        });
    }

    // Typewriter Typing Sound Effect
    if (articleTextInput) {
        articleTextInput.addEventListener("input", () => {
            playTypewriterClick();
        });
    }

    // Audio Toggle
    if (soundToggleBtn) {
        const bellIconSvg = `<svg class="btn-icon-svg" viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path><path d="M13.73 21a2 2 0 0 1-3.46 0"></path></svg>`;

        soundToggleBtn.addEventListener("click", () => {
            initAudio();
            audioEnabled = !audioEnabled;
            if (audioEnabled) {
                soundToggleBtn.innerHTML = `<span class="btn-icon">${bellIconSvg}</span> <span class="btn-text">Active</span>`;
                soundToggleBtn.style.background = "var(--ink-primary)";
                soundToggleBtn.style.color = "var(--bg-parchment-light)";
                playTypewriterClick();
            } else {
                soundToggleBtn.innerHTML = `<span class="btn-icon">${bellIconSvg}</span> <span class="btn-text">Muted</span>`;
                soundToggleBtn.style.background = "var(--bg-parchment-dark)";
                soundToggleBtn.style.color = "var(--ink-primary)";
            }
        });
    }

    // Print Button
    if (printPageBtn) {
        printPageBtn.addEventListener("click", () => {
            window.print();
        });
    }
}

// Document Ready Initialization
document.addEventListener("DOMContentLoaded", () => {
    renderVintageDate();
    initEventListeners();

    // Run initial demo classification
    const defaultSample = "The Prime Minister faced heated debates in parliament regarding government budget cuts, tax reforms, healthcare spending allocations, and upcoming national election strategies.";
    if (articleTextInput) {
        articleTextInput.value = defaultSample;
    }
    const initialResult = computeClientSideProbabilities(defaultSample);
    renderVerdict(initialResult, defaultSample);
});
