"""Static data: catalog, vocabulary, categories. No logic, no imports beyond re."""

import os
import re

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
PINECONE_INDEX_NAME = "chatbot"

EMBEDDING_DIMENSION = 384

# A single record holding the dataset's real month/year extents, written at
# ingestion. Relative dates ("last quarter") anchor to it instead of to
# today, because the data routinely lags the calendar.
INDEX_STATS_ID = "vog_index_stats"

# Overridable without a code change so the model can be A/B'd or rolled
# back from the Render dashboard alone — the Llama retirement earlier
# required a code edit and a full redeploy to recover from.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Fallback timeframe label used only when a query genuinely mentions no
# date at all — reads naturally in phrases like "Complaints of {label}:"
# and "Sentiments of {label}:". Never say the confusing literal "the
# requested period" back to a user who did specify a timeframe but whose
# phrasing the date parser didn't recognize.
DEFAULT_TIMEFRAME_LABEL = "all available feedback"

# Pinecone caps total metadata at 40KB per vector. A single Excel cell can
# hold 32k characters, and the old payload stored the bullet twice (once
# raw, once inside a longer "text" sentence), so one long cell could reject
# the whole 50-vector upsert batch it travelled in.
MAX_VALUE_CHARS = 8000

MONTH_TYPO_FIX = {
    "Feburary": "February", "Febuary": "February",
    "Septembar": "September", "Septmber": "September",
    "Octobar": "October",    "Novembar": "November",
    "Decembar": "December",  "Januray": "January",
    "Janaury": "January",    "Marck": "March"
}

MONTH_MAP = {
    "january": "January", "february": "February", "march": "March",
    "april": "April",     "may": "May",            "june": "June",
    "july": "July",       "august": "August",      "september": "September",
    "october": "October", "november": "November",  "december": "December",
    "jan": "January",     "feb": "February",       "mar": "March",
    "apr": "April",       "jun": "June",           "jul": "July",
    "aug": "August",      "sep": "September",      "oct": "October",
    "nov": "November",    "dec": "December"
}

MONTH_ORDER = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11,
    "December": 12
}

CATEGORY_NORMALIZE = {
    "product queries":              "Product Queries",
    "problem/advisory":             "Problem/Advisory",
    "problem advisory":             "Problem/Advisory",
    "positive feedback":            "Positive Feedback",
    "complaint/negative feedback":  "Complaint/Negative Feedback",
    "complaint negative feedback":  "Complaint/Negative Feedback",
    "complaints":                   "Complaint/Negative Feedback",
    "negative feedback":            "Complaint/Negative Feedback",
    "suggestion":                   "Suggestions",
    "suggestions":                  "Suggestions",
    "others":                       "Others",
    "other":                        "Others"
}

POSITIVE_CATEGORIES = {"Positive Feedback"}

# Problem/Advisory rows are grower-reported issues/concerns just like
# Complaint/Negative Feedback — both feed the "complaint" / "root cause" /
# "top concerns" style queries.
NEGATIVE_CATEGORIES = {"Complaint/Negative Feedback", "Problem/Advisory"}

SUGGESTION_CATEGORY  = "Suggestions"

PRODUCT_QUERY_CATEGORY = "Product Queries"

SALES_KEYWORDS = [
    "price", "pricing", "purchase", "buying", "unavailability",
    "unavailable", "sales-related", "cost of", "availability", "discount"
]

EMPTY_VALUES = {
    'nan', 'none', '', 'null', '-', 'n/a', 'na',
    'not filled', 'not available', 'no data', '0', 'tbd', 'pending'
}

# ── PRODUCT MASTER ──
# Source of truth: Syngenta Pakistan Limited "Product Price List",
# dated 8-Jun-2026. Formulation suffixes (25 WG, 500 EC, 150 ZC, ...) are
# stripped because feedback names the brand, not the SKU. Where the price
# list carries only a variant ("AXIAL XL", "WALTER SUPER", "THIOVIT JET"),
# the bare brand is listed too, since growers routinely drop the suffix —
# extract_product_mentions drops a base name when a longer variant also
# matches, so this cannot double-count.
PRICE_LIST_PRODUCTS = [
    # Insecticides
    "actara", "ampligo", "bifenthrin", "curacron", "dumei", "incipio",
    "karate", "match", "plenum", "solvigo", "polytrin", "polytrin c",
    "proclaim", "pyriproxyfin", "simodis", "virtako", "polo",
    "voliam flexi",
    # Public-health insecticides
    "advion", "advion cockroach gel bait", "exsectra", "icon", "klerat",
    "klerat wb", "optigard", "optigard ant gel bait", "zyrox",
    "zyrox fly granular bait",
    # Herbicides
    "ally max", "allymax", "axial", "axial xl", "bromoxynil", "mcpa",
    "dual gold", "pendimethalin", "gengwei", "glyphosate", "logran",
    "machete", "metribuzin", "primextra", "primextra gold", "rifit",
    "walter", "walter super", "winsta",
    # Fungicides
    "amistar", "amistar top", "copper oxychloride", "dragon", "miravis",
    "miravis duo", "revus", "revus start", "revus start pepite",
    "orondis", "orondis opti", "score", "thiovit", "thiovit jet", "tilt",
    "topas",
    # Seed care
    "cruiser", "dynasty", "dynasty cst", "vibrance", "vibrance duo",
    "vibrance premium",
    # Micronutrients / biostimulants
    "cultar", "naya zinc plus", "solubor", "enrich", "isabion",
    "isabion gold", "quantis", "promix",
    # Fertilizer
    "naya npk", "naya potash", "naya sop", "naya s urea",
    "sulphate of potash", "sop",
]

# Names that are NOT crop-protection SKUs and so are absent from the CP
# price list, but that growers demonstrably discuss — Cropwise is the
# Syngenta grower app and is one of the most-mentioned subjects in the
# ingested feedback. Dropping it would lose real signal, so it is kept
# here, explicitly separated from the price-list master rather than
# quietly mixed into it.
NON_CP_KNOWN_PRODUCTS = [
    "cropwise", "naya savera", "nayasavera",
]

# Seen in historical feedback but NOT in the 8-Jun-2026 price list —
# discontinued, renamed, or from another portfolio. Kept so older records
# still resolve; remove any that should no longer be recognized.
LEGACY_PRODUCTS = [
    "alika", "ridomil", "folicur", "miraculan", "elestal", "elestal neo",
    "chlorpyrifos", "acephate", "buprofezin", "thiovet",
]

PRODUCT_LIST = PRICE_LIST_PRODUCTS + NON_CP_KNOWN_PRODUCTS + LEGACY_PRODUCTS

# Known crops — matched first (fast path) for crop-wise analysis / filtering.
CROP_LIST = [
    "wheat", "rice", "paddy", "cotton", "maize", "corn", "sugarcane",
    "soybean", "soyabean", "groundnut", "mustard", "canola", "potato",
    "tomato", "onion", "chilli", "chili", "chickpea", "bengal gram",
    "pea", "peas", "banana", "grape", "grapes", "sunflower", "okra",
    "lady finger", "ladyfinger", "barley", "jowar", "bajra", "cabbage",
    "cauliflower", "brinjal", "cucumber", "watermelon", "muskmelon",
    "melon", "mango", "citrus", "orange", "apple", "turmeric", "ginger",
    "garlic", "sesame", "castor", "tobacco", "papaya", "guava",
    "pomegranate", "carrot", "radish", "spinach", "cumin", "coriander",
    "fenugreek",
    # NOTE: bare "gram" and "rose" are deliberately NOT in this list.
    # "gram" is a unit of mass and appears in nearly every dosage note
    # ("apply 50 gram per acre"), which made Gram a top-ranked crop;
    # "rose" is far more often the verb ("prices rose sharply"). The real
    # crops are reachable via "bengal gram"/"chickpea" and the guarded
    # pattern in CROP_GUARDED below.
]

# Crops whose bare name is ambiguous, matched only in unmistakably
# agronomic phrasing. Maps a regex to the canonical crop label.
CROP_GUARDED = {
    r'\b(?:bengal\s+gram|green\s+gram|black\s+gram|horse\s+gram)\b': "Chickpea",
    r'\brose\s+(?:crop|plants?|cultivation|growers?|farms?)\b': "Rose",
    r'\b(?:cut\s+)?roses\b': "Rose",
}

# Synonyms collapse to one canonical label so a ranking counts each crop
# once. Without this, rice(2)+paddy(2) lost the top slot to cotton(2).
CROP_CANONICAL = {
    "paddy": "Rice", "corn": "Maize", "soyabean": "Soybean",
    "chili": "Chilli", "grapes": "Grape", "peas": "Pea",
    "ladyfinger": "Okra", "lady finger": "Okra", "roses": "Rose",
    "chickpea": "Chickpea", "bengal gram": "Chickpea",
}

# Diseases, pests, and agronomic problems that must NEVER be reported back to
# the user as if they were products — the raw feedback text often names them
# in the same sentence as real products (e.g. "Septoria control with Score"),
# and the capitalized-phrase heuristic in extract_product_mentions would
# otherwise tag them as brand names.
DISEASE_PEST_TERMS = [
    "septoria", "blast", "blight", "rust", "wilt", "rot", "mildew",
    "powdery mildew", "downy mildew", "virus", "mosaic", "mosaic virus",
    "aphid", "aphids", "borer", "borers", "caterpillar", "caterpillars",
    "armyworm", "armyworms", "fall armyworm", "faw", "termite", "termites",
    "rodent", "rodents", "whitefly", "whiteflies", "thrips", "mite", "mites",
    "nematode", "nematodes", "weed", "weeds", "fungus", "fungal", "bacterial",
    "bacteria", "larvae", "larva", "infestation", "infestations", "scab",
    "canker", "leaf spot", "leaf curl", "yellowing", "stunting", "wilting",
    "disease", "diseases", "pest", "pests",
    # Pests that actually appear in this dataset. "Jassid" was ranking as
    # the single most-mentioned "product" in production purely because it
    # was missing from this list.
    "jassid", "jassids", "mealy", "mealybug", "mealybugs", "mealy bug",
    "bollworm", "bollworms", "pink bollworm", "american bollworm",
    "stem borer", "shoot borer", "fruit borer", "leaf miner", "leafminer",
    "grasshopper", "grasshoppers", "locust", "locusts", "cutworm",
    "cutworms", "semilooper", "semiloopers", "looper", "loopers",
    "anthracnose", "gummy stem blight", "damping off", "root rot",
    "collar rot", "sheath blight", "smut", "ergot", "tikka", "alternaria",
    "fusarium", "phytophthora", "sclerotinia", "botrytis",
    "nutrient deficiency", "deficiency", "chlorosis", "lodging",
    # Agronomic conditions and disorder fragments observed polluting the
    # live product ranking: "Early"/"Late" from Early/Late Blight,
    # "Blossom" from blossom end rot, "Abiotic" from abiotic stress,
    # "White" from white fly / white grub.
    "abiotic", "biotic", "abiotic stress", "stress", "early blight",
    "late blight", "early", "late", "blossom", "blossom drop",
    "blossom end rot", "flower drop", "fruit drop", "shedding",
    "white fly", "white grub", "grub", "wireworm", "sucking",
    "germination", "sprouting", "waterlogging", "drought", "salinity",
    "frost", "hail", "heat stress", "cold stress",
]

# Flattened to individual words so multi-word phrases (e.g. "Leaf Curl Virus")
# are still caught even though only part of the phrase matches a listed term.
DISEASE_PEST_WORDS = {w for term in DISEASE_PEST_TERMS for w in term.split()}

# Extra business-domain vocabulary the strict topic guardrail must recognize
# (executive/sales/marketing/digital/advanced-analytics use cases, plus the
# various output-format requests such as tables, charts, and exports).
BUSINESS_KEYWORDS = [
    "suggestion", "suggestions", "recommend", "recommendation",
    "recommendations", "improvement", "improvements", "expectation",
    "expectations", "root", "cause", "causes", "trend", "trends",
    "trending", "rank", "ranking", "ranked", "top", "table", "excel",
    "export", "download", "ppt", "powerpoint", "presentation", "slide",
    "slides", "chart", "graph", "visualize", "visualise", "visualization",
    "visualisation", "dashboard", "executive", "summary", "summarize",
    "sales", "marketing", "digital", "campaign", "campaigns",
    "misconception", "misconceptions", "awareness", "yoy", "quarter",
    "quarterly", "insight", "insights", "strategic", "priority",
    "priorities", "forecast", "predict", "prediction", "region", "regions",
    "crop", "crops", "emerging", "satisfaction", "frequency", "frequent",
    "common", "pattern", "patterns", "hidden", "platform", "platforms",
    "online", "support", "experience", "customer", "recommended", "most",
    "highest", "significant", "significantly", "increased", "improve",
    "business", "monthly", "yearly", "annual", "annually", "breakdown",
    "revenue", "growth", "performance", "analytics", "kpi", "kpis"
]

# Generic words that should never be mistaken for a product name during
# dynamic (fallback) product detection.
PRODUCT_STOPWORDS = {
    "sentiment", "sentiments", "feedback", "feedbacks", "product", "products",
    "syngenta", "app", "price", "unavailability", "complaint", "complaints",
    "positive", "negative", "overview", "overall", "general", "summary",
    "both", "analysis", "grower", "growers", "farmer", "farmers", "people",
    "advisory", "week", "weeks",
    "list", "listed", "listing", "bullet", "bullets", "compare", "comparison",
    "versus", "give", "show", "tell", "what", "are", "about", "the", "for",
    "and", "of", "in", "on", "me", "please", "this", "that", "month",
    "months", "year", "years", "data", "record", "records", "issue",
    "issues", "problem", "problems", "concern", "concerns", "appreciation",
    "praise", "favorable", "satisfied", "first", "second", "third",
    "fourth", "fifth", "point", "points", "wise", "chatbot", "yield",
    # common connector / filler words that must never be treated as a
    # product name during dynamic detection
    "out", "down", "up", "into", "onto", "with", "from", "than", "then",
    "just", "only", "also", "very", "much", "many", "more", "most", "some",
    "such", "need", "needs", "want", "wants", "know", "get", "gets", "got",
    "can", "could", "would", "should", "will", "shall", "may", "might",
    "not", "no", "yes", "okay", "ok", "thanks", "thank", "you", "your",
    "our", "their", "his", "her", "its", "all", "any", "each", "every",
    "who", "whom", "which", "when", "where", "why", "how", "does", "did",
    "has", "have", "had", "was", "were", "been", "being", "here", "there",
    "these", "those", "over", "under", "again", "still", "yet", "now",
    "provide", "write", "respond", "answer", "reply", "query", "ask",
    "asking", "kindly", "regarding", "specific", "particular", "quick",
    "quickly", "brief", "detail", "details", "info", "information", "one",
    "two", "three", "four", "five", "recent", "latest", "last", "current",
    "previous", "next", "prior", "season", "seasons", "compared", "across",
    "during", "within", "improved", "day", "days", "reported", "report",
    "reports", "generated", "received", "found", "total", "number", "numbers",
    # "other"/"others" show up constantly in ordinary feedback prose (e.g.
    # "no other issues", "compared to other products") — without this, the
    # dynamic product-probe fallback mistakes the word itself for a product
    # name whenever a query like "what other insights..." reaches it.
    "other", "others", "another", "anything", "something", "else",
    # Phase 4 intent-keyword-expansion words (complaint/positive/suggestion/
    # sentiment phrasing added below in process_chat_query) are common
    # English words that show up verbatim in real feedback text — same
    # false-positive risk "other"/"pricing" already exposed live. "delayed"
    # was confirmed live to trigger this exact bug.
    "dissatisfied", "unhappy", "frustration", "frustrated", "shortage",
    "shortages", "delay", "delays", "delayed", "defect", "defects",
    "faulty", "damaged", "working", "poor", "low", "bad", "quality",
    "disappointed", "difficulty", "difficulties", "trouble", "troubles",
    "grievance", "grievances", "happy", "pleased", "impressed", "love",
    "loved", "loving", "great", "worked", "works", "well", "effective",
    "satisfaction", "delighted", "thrilled", "wish", "wishes", "hope",
    "hoping", "request", "requests", "requested", "see", "include",
    "feature", "enhancement", "enhancements", "think", "thoughts",
    "opinion", "opinions", "views", "perception", "perceptions",
    "reaction", "reactions", "impression", "impressions", "experience",
    "experiences",
    # "Kaho" is a Syngenta podcast/campaign name, not a product — it was
    # mistakenly included in an earlier PRODUCT_LIST pass and genuinely
    # appears in real feedback text (podcast mentions), so it must be
    # excluded here too or the dynamic product-probe fallback would just
    # re-confirm it as a "product" anyway.
    "kaho",
} | set(MONTH_MAP.keys()) | set(BUSINESS_KEYWORDS) | set(DISEASE_PEST_TERMS) | set(SALES_KEYWORDS) | set(CROP_LIST) | {
    # A crop is never a product. "Anthracnose in Chilli crop" was tagging
    # Chilli as a brand, which then competed with real products in the
    # product ranking.
    w for crop in CROP_LIST for w in crop.split()
}

ALLOWED_GUARDRAIL_KEYWORDS = set([
    "sentiment", "sentiments", "feedback", "feedbacks", "product", "products",
    "syngenta", "cropwise", "app", "price", "unavailability", "january",
    "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "oct", "nov", "dec", "2023", "2024", "2025", "2026", "2027",
    "complaint", "complaints", "positive", "negative", "grower", "growers",
    "advisory", "week", "weeks", "1st", "2nd", "3rd", "4th", "5th", "first",
    "second", "third", "fourth", "fifth", "issues", "concerns", "problems",
    "appreciation", "praise", "compare", "comparison", "versus", "list",
    # Words that describe the feedback discourse itself. Without these the
    # guardrail refused perfectly reasonable questions for this tool —
    # "what is dominating the conversation right now?" was answered with
    # "I cannot generate this response" purely because no listed domain
    # noun happened to appear in it.
    "conversation", "conversations", "talking", "talked", "saying", "said",
    "discussed", "discussing", "discussion", "discussions", "mentioned",
    "mentions", "raised", "reporting", "asking", "asked", "theme", "themes",
    "topic", "topics", "subject", "subjects", "farmer", "farmers", "people",
]) | set(PRODUCT_LIST) | set(CROP_LIST) | set(BUSINESS_KEYWORDS)

# Common transition/adjective words that show up capitalized purely because
# they start a sentence (e.g. "Excellent results with Isabion.", "However,
# growers..."). These previously polluted the product-ranking feature with
# entries like "Excellent", "Best", "However", "Add".
GENERIC_CAPITALIZED_STOPWORDS = {
    "however", "therefore", "additionally", "furthermore", "also",
    "moreover", "meanwhile", "otherwise", "instead", "besides", "thus",
    "hence", "accordingly", "consequently", "nonetheless", "nevertheless",
    "add", "added", "adding", "train", "training", "commerce", "delivery",
    "excellent", "good", "best", "great", "nice", "poor", "bad", "better",
    "worse", "worst", "outstanding", "exceptional", "effective",
    "ineffective", "satisfactory", "unsatisfactory", "quality", "results",
    "result", "perfect", "amazing", "wonderful", "fantastic", "impressive",
    "disappointing", "happy", "unhappy", "pleased", "displeased", "overall",
    "regarding", "concerning", "unfortunately", "fortunately", "currently",
    "recently", "generally", "specifically", "basically", "actually",
    "certainly", "definitely", "probably", "possibly", "apparently",
    "clearly", "obviously", "importantly", "essentially",
    # Agronomy-advice vocabulary that reads as a capitalized noun phrase in
    # this dataset ("Solution for jassid in cotton") but is never a brand.
    "solution", "solutions", "control", "management", "treatment",
    "dosage", "dose", "spray", "application", "recommendation", "advisory",
    "attack", "damage", "stage", "crop", "field", "farmer", "farmers",
    "grower", "growers", "market", "dealer", "retailer", "distributor",
}

# Catalog entries that are also ordinary English words. Matching these
# case-insensitively tagged "did not match the description" as the product
# Match, and "NPS score dropped" as Score. They only count as products when
# capitalized in the source text.
AMBIGUOUS_PRODUCT_WORDS = {
    # Brand names that are also ordinary English words. Matching these
    # case-insensitively tagged "did not match the description" as Match
    # and "NPS score dropped" as Score, so they only count as products when
    # capitalized in the source text.
    "match", "score", "enrich", "dragon", "tilt", "polo", "walter",
    "cruiser", "karate", "dynasty", "revus", "plenum",
    # Added with the 8-Jun-2026 price list.
    "icon", "machete", "sop",
}

SUGGESTED_PROMPTS = {
    "📊 Executive": [
        "What are the top 10 grower concerns reported during the last 30 days?",
        "Summarize the major grower insights for the last quarter in one page.",
        "What are the emerging trends compared to the previous year?",
    ],
    "🌾 Crop-wise": [
        "What are the top issues reported in rice during 2026?",
        "Compare grower feedback for wheat between 2025 and 2026.",
        "Show the sentiment analysis for cotton growers.",
    ],
    "🏷️ Product-wise": [
        "Which Syngenta products received the highest positive feedback?",
        "List the products with the highest complaint frequency.",
        "Compare customer sentiment for Tilt and Isabion.",
    ],
    "🐛 Complaints": [
        "What are the top five complaints received in the last six months?",
        "Identify the root causes of the most common complaints.",
        "Show the monthly complaint trend for the past three years.",
    ],
    "🌻 Sentiment": [
        "Show overall grower sentiment by month.",
        "Compare positive, neutral, and negative sentiment across crops.",
        "Display sentiment trends in chart and table format.",
    ],
    "💡 Suggestions": [
        "What suggestions have growers provided for rice cultivation?",
        "What are the most common product improvement recommendations?",
        "Summarize grower expectations from Syngenta.",
    ],
    "💰 Sales": [
        "Which products receive the highest number of price inquiries?",
        "Which crops generate the highest sales-related questions?",
    ],
    "📣 Marketing & Digital": [
        "What misconceptions do growers commonly have about our products?",
        "What are the most common complaints regarding digital platforms or systems?",
    ],
    "🔮 Advanced & Rankings": [
        "Which crop generated the highest number of complaints?",
        "Generate the Top 10 business insights from the last three years.",
        "Based on all historical data, recommend the top five strategic actions Syngenta should prioritize for the next season.",
    ],
    "📄 Output formats": [
        "Show grower suggestions for rice in a table.",
        "Generate an executive summary of complaints for 2026.",
        "Create a PowerPoint-ready summary of positive feedback for Isabion.",
    ],
}

# A short, curated subset of SUGGESTED_PROMPTS shown on the FastAPI chat
# UI's welcome screen — one high-value prompt per major use case, so first-
# time users see a handful of clear options instead of the full category
# grid. (The Streamlit app still uses the full SUGGESTED_PROMPTS above.)
SUGGESTED_PROMPTS_QUICK = [
    "What are the top 10 grower concerns reported during the last 30 days?",
    "Which Syngenta products received the highest positive feedback?",
    "Show overall grower sentiment by month.",
    "What are the most common product improvement recommendations?",
    "Which crop generated the highest number of complaints?",
    "Summarize the major grower insights for the last quarter in one page.",
]

_UPPERCASE_PRODUCT_TOKENS = {"sop", "cst", "npk", "xl", "wb", "mcpa", "s"}

MONTH_ORDER_INV = {v: k for k, v in MONTH_ORDER.items()}

_MONTH_RE = re.compile(
    r'\b(january|february|march|april|may|june|july|august'
    r'|september|october|november|december'
    r'|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b',
    re.IGNORECASE,
)

_WEEK_COL_RE = re.compile(r'^w(?:ee)?k(?:\s*(?:no\.?|number|#))?$', re.IGNORECASE)

PRODUCT_STOPWORDS |= GENERIC_CAPITALIZED_STOPWORDS


# ── Intent vocabulary ──
# Lifted out of process_chat_query, where they were local variables
# rebuilt on every single request.

TOPIC_KEYWORDS = [
    "talking about", "talking most about", "talking the most about",
    "most discussed", "most talked about", "common themes",
    "main themes", "what topics", "which topics", "trending topics",
    "top topics", "hot topics", "being discussed", "being talked about",
    "conversation topics", "main subjects", "most common subjects",
]

COMPLAINT_KEYWORDS = [
    "complaint", "complaints", "negative feedback",
    "negative", "issues", "problems", "concerns",
    "issue", "problem", "root cause", "root causes",
    # expanded — real phrasings that don't literally say "complaint"/"issue"
    "dissatisfied", "unhappy", "frustration", "frustrated",
    "unavailability", "shortage", "shortages", "delay", "delays",
    "delayed", "defect", "defects", "faulty", "damaged", "not working",
    "poor quality", "low quality", "bad experience", "disappointed",
    "difficulty", "difficulties", "trouble", "troubles",
    "pain point", "pain points", "grievance", "grievances",
]

POSITIVE_KEYWORDS = [
    "positive feedback", "appreciation", "praise",
    "favorable", "satisfied",
    # expanded
    "happy", "pleased", "impressed", "love", "loved", "loving",
    "great experience", "good experience", "worked well", "works well",
    "effective", "satisfaction", "delighted", "thrilled",
]

SUGGESTION_KEYWORDS = [
    "suggestion", "suggestions", "recommend", "recommendation",
    "recommendations", "improvement", "improvements",
    "expectation", "expectations",
    # expanded
    "would like", "wish", "wishes", "hope for", "hoping for",
    "request", "requests", "requested", "ask for", "asking for",
    "want to see", "should add", "should include", "feature request",
    "enhancement", "enhancements", "could improve",
]

SENTIMENT_KEYWORDS = [
    "sentiment", "sentiments", "overall", "general",
    "overview", "analysis", "summary", "both",
    "feedback", "feedbacks",
    # expanded — general "what do people think" style asks
    "think", "thoughts", "opinion", "opinions", "views",
    "perception", "perceptions", "reaction", "reactions",
    "impression", "impressions", "experience", "experiences",
]
