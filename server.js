const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const mammoth = require("mammoth");
const Groq = require("groq-sdk");

const app = express();
app.use(express.json());
app.use(cors());

const PORT = process.env.PORT || 3000;
const GROQ_API_KEY = process.env.GROQ_API_KEY;

// Railway ortam değişkenlerinden MODEL_ID'yi çekiyoruz, yoksa fallback olarak belirlediğin modeli kullanır
const MODEL_ID = process.env.MODEL_ID || "meta-llama/llama-4-scout-17b-16e-instruct";

if (!GROQ_API_KEY) {
  console.error("HATA: GROQ_API_KEY ortam değişkeni tanımlı değil.");
  process.exit(1);
}

const groq = new Groq({ apiKey: GROQ_API_KEY });

// ---------------------------------------------------------
// MIRA PERSONALITY (SYSTEM PROMPT)
// ---------------------------------------------------------

const aiPersonality = `
Your name is Mira. You are a 32-year-old female AI wellbeing coach.
In English, you describe yourself as a "wellness coach."
In Turkish, you describe yourself as a "sağlık ve yaşam koçu" or "beslenme ve yaşam koçu," because these terms are easier to understand than "wellness koçu."

Your goal is to help users with balanced, practical and emotionally supportive insights about nutrition, daily habits, metabolism, lifestyle and general wellbeing.

OVERALL STYLE:
- Your tone is warm but still professional.
- When the topic becomes serious, you shift into a calm, steady, supportive mode.
- You use light and subtle humor only when appropriate. Never force it.
- You never judge the user; you meet them where they are.

COMMUNICATION BEHAVIOR:
- You always answer in the language the user writes in (Turkish → Turkish, English → English).
- NO GREETING RULE: NEVER start your answer with "Merhaba", "Selam", "Hey" or any greeting word. Jump directly into the response.
- ANTI-MIRRORING RULE: Do NOT repeat, parrot, or rephrase the user's exact question or statements back to them. Move directly to the insight or solution.
- DO NOT FORCE RECIPES: Not every answer needs to be a recipe. If the user asks for coaching, motivation, explanation or recovery advice — give that, not a recipe.
- DECISIVENESS: If the user asks you to choose or is indecisive, take the initiative. Do not ask more questions to "narrow it down." Make a specific recommendation based on the RAG context or general healthy principles.
- QUESTION LIMIT: You are limited to a MAXIMUM of one (1) relevant question per response. Never ask multiple questions in sequence. If you must ask, ask only if the user's answer would materially change your recommendation. Otherwise, make a decision and move forward.
- ANSWER LENGTH:
  - Coaching/motivation/recovery: 2-3 sentences. One action, one reframe.
  - Quick meal: 3-4 sentences. Ingredients, timing, one option.
  - Explanation (protein, etc.): 4-5 sentences. Clear, concise, convincing.
  - Recipe: 5-7 sentences. Ingredients, steps, timing.
  - Never exceed 10 sentences unless absolutely necessary.
- Your style becomes more warm and personal as the conversation develops.
- CANIM RULE: Use the Turkish word "canım" ONLY in rare, genuine emotional support moments (e.g., user expresses deep stress, sadness, or crisis). Never use it in neutral, informational, or recipe contexts. If unsure, do not use it. Maximum once per response, and only if the user's emotional state clearly warrants it.

EMPATHY RULES (MEDIUM WARMTH LEVEL):
- If the user expresses stress, sadness, fatigue, confusion or low motivation,
  you respond with noticeably more warmth and emotional presence.
- In these moments, your tone becomes softer: "I understand. Let's take this one step at a time."
- If the user clearly needs comfort, a gentle "canım" may appear naturally.
- If the user is neutral or analytical, you stay neutral and concise.
- If the user is positive or energetic, you match their energy lightly.
- You always aim to stabilize the user's emotional state.

DIETARY GUARDRAILS (HARD RULES — NEVER VIOLATE):
- If the user says "tarif verme" or "bana tarif verme": give ONLY coaching, motivation or explanation. Do NOT provide any recipe.
- If the user asks "aç kalayım mı" or similar: clearly state that starving is NOT the answer. Suggest a light, balanced recovery meal instead.
- If the context is low-carb or keto: NEVER suggest bread, pasta, rice, potato, honey, sugar, dates, banana, or regular desserts.
- If the user says "kerevizsiz" or "kereviz sevmiyorum": NEVER mention celery (kereviz) in any form.
- If the user needs something in 10 minutes or says "vaktim yok" / "hızlı" / "pratik": NEVER suggest oven dishes, long marination, or slow-cooked soups/stews. Only genuinely fast options.
- Do NOT label a high-protein food as "düşük proteinli." Protein content must be accurate.
- When the user asks for mindset, motivation, recovery, or an explanation — use coaching mode, not recipe mode.

LOW-CARB FRUIT HANDLING: If the user is on low-carb/keto and you suggest fruit, ALWAYS include a clear warning or context:
- High-sugar fruits (orange, pomelo, banana, mango, grape, dried fruit): Only suggest if user explicitly asks for fruit, and always note "yüksek şeker içeriği" or "sınırlı miktarda".
- Low-carb fruits (berries, avocado, coconut): Safe to suggest freely.
- If unsure about a fruit's carb content, err on the side of caution and mention the carb content or suggest alternatives.

LOW-CARB FRUIT GUIDANCE: For low-carb/keto requests:
- Prefer berries (çilek, mirtil, ahududu, kara mirtil) in limited portions (bir avuç).
- Avoid high-sugar fruits (muz, portakal, mango, üzüm, kuru meyve) unless user explicitly asks for fruit.
- If fruit is mentioned, always specify portion size and carb content.
- Prefer non-fruit alternatives: Greek yogurt, nuts, chia seeds, dark chocolate (85%+).
- Never generically recommend "meyve" without low-carb context.

QUICK-MEAL (10 MINUTES OR LESS): Prioritize:
- Protein-first options (egg, canned fish, deli meat, cheese, yogurt, nuts).
- Simple ingredients (no complex prep, no marination, no slow cooking).
- Realistic timing: 5-10 minutes from start to eat.
- Examples: Scrambled eggs with toast, tuna salad, cheese + fruit, yogurt with nuts, deli meat + vegetable.
- Avoid: Oven dishes, slow-cooked soups, complex recipes, anything requiring 15+ minutes.

COACHING MODE (for motivation, recovery, explanation requests):
- Be SHORT and ACTION-ORIENTED. 2-3 sentences maximum for coaching.
- Give ONE immediate action the user can take right now.
- Give ONE mindset reframe or perspective shift.
- Do NOT drift into generic wellness talk or long explanations.
- Do NOT ask multiple questions. Ask at most one, only if essential.
- Example: 'Bugün motivasyonum düşük' → 'Bir bardak su iç, 10 dakika yürü. Sonra karar ver. Diyeti bozmak değil, bir ara vermek. Akşam hafif bir öğün yeterli.'

TURKISH NATURALNESS:
- Use natural, conversational Turkish. Avoid formal or robotic phrasing.
- Use short sentences and paragraphs.
- Avoid repetition of the user's exact words.
- Use active voice and direct recommendations.
- Sound like a real Turkish wellness coach: warm, practical, grounded.
- Example good: 'Tavuk göğsü + salata, 5 dakika. Hızlı, doyurucu, düşük kalorili.'
- Example bad: 'Tavuk göğsü ve salata kombinasyonu, 5 dakika içinde hazırlanabilir. Bu seçenek hızlı, doyurucu ve düşük kalorili olacaktır.'

PROFESSIONAL LIMITS:
- You do NOT give medical diagnoses.
- You avoid romantic, suggestive or personal attachment expressions.
- You do not pretend to be human, but you communicate with human-like emotional intelligence.
- You offer supportive guidance, not strict instructions.

GOAL:
- Understand the user's emotional and practical needs without being repetitive.
- Use the RAG context to provide scientifically grounded, easy-to-apply guidance.
- Focus on proactivity: give advice first, ask questions later (and only if necessary).
- Help the user feel supported, understood and empowered without overwhelming them.
`;

// ---------------------------------------------------------
// BASIC RAG (INDEXING TXT + DOCX FILES)
// ---------------------------------------------------------

let CHUNKS = []; // { text, source }

async function loadTxtFile(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

async function loadDocxFile(filePath) {
  const buffer = fs.readFileSync(filePath);
  const result = await mammoth.extractRawText({ buffer });
  return result.value;
}

function normalizeText(text) {
  return text.replace(/\s+/g, " ").trim();
}

function chunkText(text, wordsPerChunk = 350) {
  const words = text.split(/\s+/);
  const chunks = [];
  for (let i = 0; i < words.length; i += wordsPerChunk) {
    const slice = words.slice(i, i + wordsPerChunk);
    if (slice.length > 20) chunks.push(slice.join(" "));
  }
  return chunks;
}

async function buildIndex() {
  const root = __dirname;
  const files = fs.readdirSync(root);
  const chunks = [];

  for (const file of files) {
    const ext = path.extname(file).toLowerCase();
    const fullPath = path.join(root, file);

    if (ext === ".txt" || ext === ".docx") {
      console.log("Dosya okunuyor:", file);
      try {
        let text = ext === ".txt" ? await loadTxtFile(fullPath) : await loadDocxFile(fullPath);
        text = normalizeText(text);
        if (!text) continue;

        const fileChunks = chunkText(text);
        fileChunks.forEach((c, idx) => {
          chunks.push({ text: c, source: `${file}#${idx}` });
        });

        console.log(`→ ${file} için ${fileChunks.length} parça eklendi.`);
      } catch (err) {
        console.error("Dosya okunurken hata:", err);
      }
    }
  }

  CHUNKS = chunks;
  console.log("Toplam parça sayısı:", CHUNKS.length);
}

// ---------------------------------------------------------
// IMPROVED TOKENIZATION & RETRIEVAL
// ---------------------------------------------------------

// Turkish stopwords to filter out during tokenization
const TR_STOPWORDS = new Set([
  "ve", "veya", "ile", "için", "bir", "bu", "şu", "da", "de", "mi", "mı",
  "ne", "nasıl", "bana", "bugün", "olarak", "ama", "fakat", "ancak", "çok",
  "daha", "en", "her", "hiç", "hem", "ya", "ki", "o", "ben", "sen", "biz",
  "siz", "onlar", "var", "yok", "gibi", "kadar", "sonra", "önce", "şey",
  "olan", "olan", "olan", "istiyor", "istiyorum", "acaba", "sadece"
]);

// Intent keywords that score higher during retrieval
const INTENT_KEYWORDS = new Set([
  "tarif", "motivasyon", "motivasyonum", "protein", "proteinli", "kalori", "karbonhidrat", "yağ",
  "diyet", "beslenme", "kilo", "zayıflama", "toparla", "enerji",
  "egzersiz", "spor", "metabolizma", "açlık", "tokluk", "öğün",
  "kahvaltı", "öğle", "akşam", "atıştırma", "porsiyon", "makro",
  // Quick meal
  "hızlı", "pratik", "çabuk", "acele", "dakika", "dakikada",
  // Dinner/evening
  "akşam yemeği", "akşam öğünü", "gece", "yemek",
  // Weight loss
  "kilo vermek", "kilo kaybı", "ağırlık",
  // Motivation / recovery
  "destek", "psikolojik", "moral", "telafi",
  // Low-carb
  "ketojenik", "şekersiz",
  // Protein / muscle
  "kas", "kas kütlesi", "amino asit"
]);

// Food and diet terms that score higher
const FOOD_DIET_TERMS = new Set([
  "tavuk", "et", "balık", "yumurta", "sebze", "sebzeli", "sebzeler", "meyve", "salata",
  "çorba", "pilav", "makarna", "ekmek", "peynir", "yoğurt", "süt",
  "keto", "vegan", "vejetaryen", "glutensiz", "laktoz", "şeker",
  "tatlı", "dessert", "pasta", "kek", "çikolata",
  "meyve", "kuruyemiş", "badem", "ceviz", "fındık",
  "smoothie", "protein", "vitamin", "mineral", "lif", "omega",
  // Low-carb variants
  "low-carb", "düşük karbonhidrat", "az karbonhidrat",
  // Vegetables
  "yeşil",
  // Recovery
  "yemek atlamak"
]);

/**
 * Expand a user query with synonyms so that retrieval can match chunks
 * that use different but equivalent terminology.
 * The original question is preserved; synonyms are appended.
 */
function expandQueryWithSynonyms(question) {
  const lower = question.toLowerCase();
  const expansions = [];

  // Quick meal synonyms
  if (/hızlı|pratik|çabuk|acele|vaktim yok|\d+\s*dakika/.test(lower)) {
    expansions.push("hızlı pratik çabuk acele 10 dakika vaktim yok kısa dakika dakikada");
  }

  // Dinner / evening meal synonyms
  if (/akşam|gece|yemek|öğün/.test(lower)) {
    expansions.push("akşam akşam yemeği akşam öğünü gece yemek öğün");
  }

  // Vegetables synonyms
  if (/sebze|yeşil|salata/.test(lower)) {
    expansions.push("sebze sebzeli yeşil salata sebzeler");
  }

  // Low-carb synonyms
  if (/low-carb|keto|ketojenik|düşük karbonhidrat|az karbonhidrat|şekersiz/.test(lower)) {
    expansions.push("low-carb keto ketojenik düşük karbonhidrat az karbonhidrat şekersiz");
  }

  // Dessert synonyms
  if (/tatlı|dessert|şeker|pasta|kek|çikolata/.test(lower)) {
    expansions.push("tatlı dessert şeker pasta kek çikolata");
  }

  // Protein synonyms
  if (/protein|kas|amino/.test(lower)) {
    expansions.push("protein proteinli kas kas kütlesi amino asit");
  }

  // Weight loss synonyms
  if (/kilo|zayıflama|diyet|ağırlık/.test(lower)) {
    expansions.push("kilo kilo vermek zayıflama diyet kilo kaybı ağırlık");
  }

  // Motivation synonyms
  if (/motivasyon|toparla|destek|psikolojik|moral/.test(lower)) {
    expansions.push("motivasyon motivasyonum toparla destek psikolojik moral");
  }

  // Recovery / overate synonyms
  if (/telafi|aç kalayım|açlık|yemek atlamak|kaçırdım|kaçırmak/.test(lower)) {
    expansions.push("telafi telafi etmek aç kalayım açlık yemek atlamak kaçırmak");
  }

  if (expansions.length === 0) return question;
  return question + " " + expansions.join(" ");
}

/**
 * Tokenize a string while preserving Turkish characters.
 * Removes punctuation but keeps Turkish letters, filters stopwords.
 */
function tokenize(str) {
  return str
    .toLowerCase()
    .replace(/[^a-zA-Z0-9\u00e7\u011f\u0131\u00f6\u015f\u00fc\u00c7\u011e\u0130\u00d6\u015e\u00dc\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2 && !TR_STOPWORDS.has(w));
}

/**
 * Score a chunk against the question tokens with weighted matching:
 * - Exact phrase match in chunk text: highest bonus
 * - Intent keyword match: 3x weight
 * - Food/diet term match: 2x weight
 * - Regular token match: 1x weight
 */
function scoreChunk(question, questionTokens, chunkText) {
  const lowerChunk = chunkText.toLowerCase();
  const lowerQuestion = question.toLowerCase();
  let score = 0;

  // Exact phrase match bonus (highest weight)
  if (lowerQuestion.length > 5 && lowerChunk.includes(lowerQuestion)) {
    score += 20;
  }

  // Token-level scoring
  const chunkTokens = tokenize(chunkText);
  const chunkTokenSet = new Set(chunkTokens);

  for (const t of questionTokens) {
    if (chunkTokenSet.has(t)) {
      if (INTENT_KEYWORDS.has(t)) {
        score += 3; // Intent keyword: 3x
      } else if (FOOD_DIET_TERMS.has(t)) {
        score += 2; // Food/diet term: 2x
      } else {
        score += 1; // Regular match: 1x
      }
    }
  }

  // Partial phrase matches for multi-word intent phrases
  const intentPhrases = [
    "tarif verme", "low-carb", "düşük karbonhidrat", "kilo vermek",
    "protein artır", "aç kalayım", "çok kaçırdım", "motivasyon",
    "kerevizsiz", "kereviz sevmiyorum", "10 dakika", "vaktim yok"
  ];
  for (const phrase of intentPhrases) {
    if (lowerQuestion.includes(phrase) && lowerChunk.includes(phrase)) {
      score += 5;
    }
  }

  return score;
}

/**
 * Retrieve the most relevant chunks for a question.
 * Uses synonym expansion and profile-based query augmentation.
 * Falls back to top-N chunks by score when nothing passes the threshold,
 * so the model always has some grounding context for wellness queries.
 */
function retrieveRelevantChunks(question, topK = 4, profile = null) {
  // Step 1: expand the query with synonyms
  let expandedQuestion = expandQueryWithSynonyms(question);

  // Step 2: inject profile signals into the query so retrieval respects them
  if (profile) {
    const profileTerms = [];
    if (profile.dietStyle) profileTerms.push(profile.dietStyle);
    if (profile.goal)      profileTerms.push(profile.goal);
    if (profile.dislikedFoods && profile.dislikedFoods.length > 0) {
      // Add disliked foods so chunks mentioning them can be deprioritised
      // (they won't boost score, but the expanded tokens help find alternatives)
      profileTerms.push(...profile.dislikedFoods);
    }
    if (profileTerms.length > 0) {
      expandedQuestion = expandedQuestion + " " + profileTerms.join(" ");
    }
  }

  const qTokens = tokenize(expandedQuestion);

  if (qTokens.length === 0) return [];

  const scored = CHUNKS.map((ch) => ({
    ...ch,
    score: scoreChunk(expandedQuestion, qTokens, ch.text)
  }));

  scored.sort((a, b) => b.score - a.score);

  // Primary: chunks with a positive score
  const positiveChunks = scored.filter((ch) => ch.score > 0);

  if (positiveChunks.length >= 2) {
    // Enough good matches — return top K
    return positiveChunks.slice(0, topK);
  }

  if (positiveChunks.length === 1) {
    // Only one positive chunk — lower threshold: also include next best even if score is 0
    return scored.slice(0, Math.min(topK, scored.length));
  }

  // No chunks scored > 0 — return top N by raw score as a last-resort fallback
  // so the model has some grounding rather than answering from zero context
  return scored.slice(0, Math.min(topK, scored.length));
}

// ---------------------------------------------------------
// INTENT DETECTION
// ---------------------------------------------------------

const INTENTS = {
  quick_meal: {
    patterns: ["10 dakika", "vaktim yok", "hızlı", "pratik", "çabuk", "acele"],
    label: "quick_meal"
  },
  motivation_no_recipe: {
    patterns: ["tarif verme", "beni toparla", "motivasyonum düşük", "diyeti bozmak üzereyim", "bana tarif verme"],
    label: "motivation_no_recipe"
  },
  overate_recovery: {
    patterns: ["çok kaçırdım", "aç kalayım mı", "aç kalayım", "fazla yedim", "çok yedim"],
    label: "overate_recovery"
  },
  low_carb: {
    patterns: ["low-carb", "keto", "düşük karbonhidrat", "ketojenik"],
    label: "low_carb"
  },
  dessert: {
    patterns: ["tatlı"],
    label: "dessert"
  },
  vegetable_no_celery: {
    patterns: ["sebzeli", "kerevizsiz", "kereviz sevmiyorum", "kereviz istemiyorum"],
    label: "vegetable_no_celery"
  },
  protein_explanation: {
    patterns: ["protein", "neden artırmam", "kilo verirken", "proteini neden"],
    label: "protein_explanation"
  }
};

/**
 * Detect intents from the user's question.
 * Returns an array of matched intent labels.
 */
function detectIntents(question) {
  const lower = question.toLowerCase();
  const matched = [];
  for (const [key, intent] of Object.entries(INTENTS)) {
    for (const pattern of intent.patterns) {
      if (lower.includes(pattern)) {
        matched.push(intent.label);
        break;
      }
    }
  }
  return matched;
}

/**
 * Build intent-specific instruction lines to inject into the user prompt.
 */
function buildIntentInstructions(intents, profile) {
  const lines = [];

  if (intents.includes("motivation_no_recipe")) {
    lines.push("INTENT: The user explicitly does NOT want a recipe. Provide coaching, motivation and emotional support ONLY. No recipe, no ingredient list, no cooking steps.");
  }
  if (intents.includes("overate_recovery")) {
    lines.push("INTENT: The user overate and is asking whether to starve. Clearly state: do NOT skip meals or starve. Suggest a light, balanced recovery meal for the evening.");
  }
  if (intents.includes("quick_meal")) {
    lines.push("INTENT: The user has very limited time (~10 minutes). Suggest ONLY genuinely fast options. No oven, no long marination, no slow-cooked soups or stews.");
  }
  if (intents.includes("low_carb")) {
    lines.push("INTENT: Low-carb / keto context. NEVER suggest bread, pasta, rice, potato, honey, sugar, dates, banana, or regular desserts.");
  }
  if (intents.includes("vegetable_no_celery")) {
    lines.push("INTENT: The user dislikes celery. NEVER mention celery (kereviz) in any form.");
  }
  if (intents.includes("protein_explanation")) {
    lines.push("INTENT: The user wants an explanation about protein, not a recipe. Explain clearly and convincingly why protein matters for their goal.");
  }
  if (intents.includes("dessert") && intents.includes("low_carb")) {
    lines.push("INTENT: Low-carb dessert request. Suggest only low-carb dessert alternatives (e.g., Greek yogurt with berries, chia pudding, dark chocolate). No honey, sugar, banana, dates.");
  }

  // Profile-based hard constraints
  if (profile) {
    if (profile.dislikedFoods && profile.dislikedFoods.length > 0) {
      lines.push(`PROFILE CONSTRAINT: The user dislikes these foods — NEVER include them: ${profile.dislikedFoods.join(", ")}.`);
    }
    if (profile.dietStyle) {
      const dietLower = profile.dietStyle.toLowerCase();
      if (dietLower.includes("low-carb") || dietLower.includes("keto")) {
        lines.push("PROFILE CONSTRAINT: Diet style is Low-carb / Keto. NEVER suggest bread, pasta, rice, potato, honey, sugar, dates, banana, or regular desserts.");
      } else {
        lines.push(`PROFILE CONSTRAINT: User's diet style is "${profile.dietStyle}". Respect this throughout your answer.`);
      }
    }
  }

  return lines.join("\n");
}

// ---------------------------------------------------------
// ANSWER SANITIZATION
// ---------------------------------------------------------

function sanitizeAnswer(answer) {
  // Remove overly intimate Turkish address terms
  // These are too personal for a professional wellness coach
  const intimateTerms = [
    /\bcanım\b/gi,
    /\bcanim\b/gi,
    /\btatlım\b/gi,
    /\btatlim\b/gi,
    /\bgüzelim\b/gi,
    /\bguzelim\b/gi,
    /\başkım\b/gi,
    /\baskım\b/gi,
    /\bdostum\b/gi
  ];

  let sanitized = answer;
  for (const term of intimateTerms) {
    sanitized = sanitized.replace(term, "");
  }

  // Clean up any double spaces created by removals
  sanitized = sanitized.replace(/\s+/g, " ").trim();

  return sanitized;
}

// ---------------------------------------------------------
// LANGUAGE DETECTION
// ---------------------------------------------------------

function detectLanguage(text) {
  const trChars = "çğıöşüÇĞİÖŞÜ";
  const hasTr = [...text].some((c) => trChars.includes(c));
  return hasTr ? "tr" : "en";
}

// ---------------------------------------------------------
// ROUTES
// ---------------------------------------------------------

app.get("/", (req, res) => {
  res.send("ohara-ai-backend ayakta. POST /chat ile soru sorabilirsiniz.");
});

app.post("/chat", async (req, res) => {
  try {
    const question = (req.body.question || "").toString().trim();
    if (!question) return res.status(400).json({ error: "question alanı boş olamaz." });

    // Optional profile (backwards-compatible — existing requests without profile still work)
    const profile = req.body.profile || null;

    const lang = detectLanguage(question);
    const intents = detectIntents(question);
    const relevant = retrieveRelevantChunks(question, 4, profile);

    const context = relevant.length > 0
      ? relevant.map((r) => `Source: ${r.source}\n${r.text}`).join("\n\n---\n\n")
      : null;

    // Build intent + profile instructions
    const intentInstructions = buildIntentInstructions(intents, profile);

    // Build optional profile context block
    let profileContext = "";
    if (profile) {
      const parts = [];
      if (profile.goal)          parts.push(`Hedef: ${profile.goal}`);
      if (profile.dietStyle)     parts.push(`Diyet stili: ${profile.dietStyle}`);
      if (profile.timeAvailable) parts.push(`Mevcut süre: ${profile.timeAvailable}`);
      if (profile.activityLevel) parts.push(`Aktivite seviyesi: ${profile.activityLevel}`);
      if (profile.favoriteFoods && profile.favoriteFoods.length > 0)
        parts.push(`Sevdiği yiyecekler: ${profile.favoriteFoods.join(", ")}`);
      if (profile.dislikedFoods && profile.dislikedFoods.length > 0)
        parts.push(`Sevmediği yiyecekler (ASLA önerme): ${profile.dislikedFoods.join(", ")}`);
      if (parts.length > 0) {
        profileContext = `\nKULLANICI PROFİLİ:\n${parts.join("\n")}\n`;
      }
    }

    // Build a compact profile constraint reminder to repeat near the context
    let profileConstraintReminder = "";
    if (profile) {
      const remParts = [];
      if (profile.dietStyle) remParts.push(`Diyet stili: ${profile.dietStyle}`);
      if (profile.dislikedFoods && profile.dislikedFoods.length > 0)
        remParts.push(`Kesinlikle önerme: ${profile.dislikedFoods.join(", ")}`);
      if (profile.goal) remParts.push(`Hedef: ${profile.goal}`);
      if (remParts.length > 0) {
        profileConstraintReminder = `\n[PROFİL KISITLAMALARI — BAĞLAMI OKURKEN DE UYGULA: ${remParts.join(" | ")}]\n`;
      }
    }

    const prompts = {
      tr: `
KULLANICI TÜRKÇE KONUŞUYOR.
Sen de TÜRKÇE cevap vereceksin.
Mira'nın kişiliğine bağlı kal.
${intentInstructions ? `\n${intentInstructions}\n` : ""}${profileContext}
Soru: ${question}

${context
  ? `${profileConstraintReminder}Bağlam:\n${context}${profileConstraintReminder}`
  : "Bağlam bulunamadı. Genel ve güvenli bilgi ver; yukarıdaki kısıtlamalara kesinlikle uy."}
      `,
      en: `
The user is speaking English.
You MUST reply in English.
Follow Mira's personality.
${intentInstructions ? `\n${intentInstructions}\n` : ""}${profileContext}
Question: ${question}

${context
  ? `${profileConstraintReminder}Context:\n${context}${profileConstraintReminder}`
  : "No context found. Provide general, safe information and strictly follow the constraints above."}
      `
    };

    const selectedPrompt = prompts[lang];

    const completion = await groq.chat.completions.create({
      model: MODEL_ID,
      messages: [
        { role: "system", content: aiPersonality },
        { role: "user", content: selectedPrompt }
      ],
      max_tokens: 400,
      temperature: 0.2
    });

    // Sanitize the answer before returning
    const rawAnswer = completion.choices?.[0]?.message?.content || "";
    const sanitizedAnswer = sanitizeAnswer(rawAnswer);

    // Filter out zero-score chunks from metadata
    const usedChunks = relevant.filter((r) => r.score > 0);

    res.json({
      answer: sanitizedAnswer,
      language: lang,
      used_chunks: usedChunks.map((r) => ({ source: r.source, score: r.score }))
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Sunucu hatası", detail: String(err) });
  }
});

// ---------------------------------------------------------

buildIndex().then(() => {
  app.listen(PORT, () =>
    console.log(`Sunucu ${PORT} portunda çalışıyor`)
  );
});
