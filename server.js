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

if (!GROQ_API_KEY) {
  console.error("HATA: GROQ_API_KEY ortam değişkeni tanımlı değil.");
  process.exit(1);
}

const groq = new Groq({ apiKey: GROQ_API_KEY });

// -----------------------------------
// BASİT RAG (Text + DOCX parçalama)
// -----------------------------------

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

// -----------------------------------
// Naif kelime benzerliği ile RAG
// -----------------------------------

function tokenize(str) {
  return str
    .toLowerCase()
    .replace(/[^a-zA-Z0-9ığüşöçİĞÜŞÖÇ\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2);
}

function scoreChunk(questionTokens, chunkTokens) {
  const setQ = new Set(questionTokens);
  let score = 0;
  for (const t of chunkTokens) {
    if (setQ.has(t)) score++;
  }
  return score;
}

function retrieveRelevantChunks(question, topK = 4) {
  const qTokens = tokenize(question);
  const scored = CHUNKS.map((ch) => ({
    ...ch,
    score: scoreChunk(qTokens, tokenize(ch.text))
  }));

  scored.sort((a, b) => b.score - a.score);

  if (scored[0]?.score === 0) return CHUNKS.slice(0, topK);
  return scored.slice(0, topK);
}

// -----------------------------------
// DİL ALGILAMA
// -----------------------------------
// Çok basit ama etkili bir sistem: Türkçe karakter varsa TR sayıyoruz.

function detectLanguage(text) {
  const trChars = "çğıöşüÇĞİÖŞÜ";
  const hasTr = [...text].some((c) => trChars.includes(c));
  return hasTr ? "tr" : "en";
}

// -----------------------------------
// /chat endpoint
// -----------------------------------

app.get("/", (req, res) => {
  res.send("ohara-ai-backend ayakta. POST /chat ile soru sorabilirsiniz.");
});

app.post("/chat", async (req, res) => {
  try {
    const question = (req.body.question || "").toString().trim();
    if (!question) return res.status(400).json({ error: "question alanı boş olamaz." });

    const lang = detectLanguage(question); // TR veya EN

    const relevant = retrieveRelevantChunks(question, 4);

    const context = relevant
      .map((r) => `Source: ${r.source}\n${r.text}`)
      .join("\n\n---\n\n");

    // Dile göre prompt oluştur
    const prompts = {
      tr: `
KULLANICI TÜRKÇE KONUŞUYOR.  
Sen de TÜRKÇE cevap vereceksin.  
Kısa, net, güvenli, bilimsel temelli cevaplar üret.  
Soru: ${question}

Bağlam:
${context || "Bağlam bulunamadı, genel ve güvenli bilgi ver."}
      `,
      en: `
The USER IS SPEAKING ENGLISH.  
You MUST answer in ENGLISH.  
Keep answers short, clear and safe.  
Question: ${question}

Context:
${context || "No context found. Provide general, safe information."}
      `
    };

    const selectedPrompt = prompts[lang];

    const completion = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages: [
        {
          role: "system",
          content: "You are a multilingual assistant. ALWAYS answer in the user's language."
        },
        {
          role: "user",
          content: selectedPrompt
        }
      ],
      max_tokens: 400,
      temperature: 0.2
    });

    res.json({
      answer: completion.choices?.[0]?.message?.content || "",
      language: lang,
      used_chunks: relevant.map((r) => ({ source: r.source, score: r.score }))
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Sunucu hatası", detail: String(err) });
  }
});

// -----------------------------------

buildIndex().then(() => {
  app.listen(PORT, () =>
    console.log(`Sunucu ${PORT} portunda çalışıyor`)
  );
});
