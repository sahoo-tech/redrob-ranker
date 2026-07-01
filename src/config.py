from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent

ARTIFACTS_DIR = str(_PROJECT_ROOT / "artifacts")
EMBEDDINGS_PATH = str(_PROJECT_ROOT / "artifacts" / "embeddings.npy")
CANDIDATE_IDS_PATH = str(_PROJECT_ROOT / "artifacts" / "candidate_ids.npy")
FEATURES_PATH = str(_PROJECT_ROOT / "artifacts" / "features.pkl")
JD_EMBEDDING_PATH = str(_PROJECT_ROOT / "artifacts" / "jd_embedding.npy")
MODEL_DIR = str(_PROJECT_ROOT / "artifacts" / "model")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_DIM = 384

JD_TEXT = """
Senior AI Engineer at Redrob AI (Series A AI-native talent intelligence platform).
Location: Pune or Noida, India (Hybrid). 5-9 years of experience.

Requires: Production experience with embeddings-based retrieval systems,
vector databases (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, FAISS),
strong Python, evaluation frameworks for ranking systems (NDCG, MRR, MAP),
hybrid search, LLM reranking, A/B testing, offline benchmarks.

Preferred: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank (XGBoost),
HR-tech experience, distributed systems, open-source contributions in AI/ML.

Looking for: Engineers at product companies (SaaS, startups), NOT consulting firms.
Must have shipped end-to-end ranking, search, or recommendation systems to real users.
Must write production code. Scrappy product-engineering attitude required.
"""

REQUIRED_SKILLS = [
    "embeddings", "vector database", "semantic search", "retrieval", "ranking",
    "sentence-transformers", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "hybrid search", "ndcg", "mrr", "map",
    "a/b testing", "python", "evaluation framework", "reranking",
    "learning to rank", "rag", "dense retrieval", "bm25", "approximate nearest neighbor",
    "information retrieval", "recommendation system", "ann",
    "vector store", "knn", "bi-encoder", "cross-encoder", "two-tower",
    "sparse retrieval", "inverted index", "tf-idf", "colbert",
    "recall@k", "mrr@k", "text embeddings", "sentence embeddings",
]

NICE_TO_HAVE_SKILLS = [
    "lora", "qlora", "peft", "fine-tuning", "fine-tuning llms", "xgboost",
    "open-source", "transformer", "langchain", "llm", "llms",
    "machine learning", "deep learning", "nlp", "pytorch", "tensorflow",
    "huggingface", "bert", "gpt", "t5", "encoder", "decoder",
    "attention mechanism", "finetuning",
]

DISQUALIFIER_TITLES = [
    "marketing manager", "operations manager", "accountant", "customer support",
    "mechanical engineer", "civil engineer", "content writer", "business analyst",
    "hr manager", "sales manager", "sales executive", "graphic designer",
    "seo specialist", "finance manager", "supply chain", "logistics",
    "project manager", "program manager", "product manager",
    "scrum master", "agile coach", "ux designer", "ui designer", "data entry",
]

AI_ENGINEER_TITLES = [
    "machine learning engineer", "ml engineer", "ai engineer", "data scientist",
    "nlp engineer", "research engineer", "applied scientist", "deep learning engineer",
    "computer vision engineer", "search engineer", "ranking engineer",
    "recommendation engineer", "senior ai engineer", "principal ai engineer",
    "software engineer", "backend engineer", "full stack engineer",
    "data engineer", "analytics engineer", "platform engineer",
    "ml infrastructure engineer", "ml platform engineer", "mlops engineer",
    "applied ml engineer", "search scientist", "ranking scientist",
    "recommendation scientist", "ml infra engineer", "staff engineer",
]

CONSULTING_FIRMS = [
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl",
    "tech mahindra", "mphasis", "hexaware", "ltimindtree", "mindtree",
    "birlasoft", "zensar", "niit technologies", "mastech", "kforce",
    "igate", "patni", "cyient",
]

FICTIONAL_COMPANIES = [
    "wayne enterprises", "initech", "pied piper", "globex", "acme corp",
    "dunder mifflin", "hooli", "stark industries", "umbrella corporation",
    "cyberdyne systems", "soylent corp", "massive dynamic",
]

PRODUCT_COMPANIES = [
    "swiggy", "zomato", "flipkart", "meesho", "cred", "razorpay", "zepto",
    "blinkit", "phonepe", "paytm", "ola", "nykaa", "sharechat", "moj",
    "dream11", "unacademy", "byju", "groww", "slice", "jupiter",
    "setu", "khatabook", "ofbusiness", "delhivery", "licious",
]

PREFERRED_LOCATIONS = [
    "pune", "noida", "delhi", "gurgaon", "gurugram", "hyderabad",
    "mumbai", "bangalore", "bengaluru", "ncr", "delhi ncr", "new delhi",
    "greater noida",
]

TIER_1_INSTITUTIONS = [
    "iit", "iit bombay", "iit delhi", "iit madras", "iit kanpur",
    "iit kharagpur", "iit roorkee", "iit guwahati", "iit hyderabad",
    "iit bhu", "iisc", "indian institute of science",
    "nit trichy", "nit surathkal", "nit warangal", "nit calicut",
    "bits pilani", "bits hyderabad", "bits goa",
    "iiit hyderabad", "iiit bangalore",
]

RELEVANT_FIELDS_OF_STUDY = [
    "computer science", "artificial intelligence", "machine learning",
    "data science", "information technology", "computer engineering",
    "electrical engineering", "mathematics", "statistics",
    "computational intelligence",
]

WEIGHTS = {
    "semantic_similarity": 0.40,
    "skill_score":         0.30,
    "career_quality":      0.30,
}

BEHAVIORAL_WEIGHTS = {
    "availability":    0.30,
    "responsiveness":  0.25,
    "commitment":      0.20,
    "engagement":      0.15,
    "notice":          0.10,
}

EXP_BANDS = [
    (0,   3,  0.15),
    (3,   5,  0.55),
    (5,   9,  1.00),
    (9,  12,  0.75),
    (12, 50,  0.50),
]

CONSULTING_ONLY_MULTIPLIER = 0.20
TITLE_MISMATCH_MULTIPLIER  = 0.25

PRODUCTION_SIGNALS = [
    "deployed", "deployment", "serving", "inference", "production", "latency",
    "a/b test", "a/b testing", "scale", "scaled", "pipeline", "real-time",
    "online", "live traffic", "serving infra", "model serving",
    "shipped", "launched", "rollout", "productionized", "qps", "throughput",
    "end-to-end", "inference pipeline", "production model", "real time serving",
]

RETRIEVAL_SIGNALS = [
    "retrieval", "embedding", "embeddings", "ranking", "reranking", "re-ranking",
    "vector", "vector store", "vector db", "index", "indexing",
    "similarity search", "ann", "approximate nearest", "semantic search",
    "dense retrieval", "sparse retrieval", "hybrid search", "bm25",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "vector search", "knn search", "nearest neighbor", "bi-encoder", "cross-encoder",
    "reranker", "embedding search", "dense vector", "inverted index",
    "passage retrieval", "information retrieval", "semantic similarity",
]

ENGINEERING_SIGNALS = [
    "architecture", "microservice", "api", "rest", "grpc", "docker",
    "kubernetes", "ci/cd", "distributed", "kafka", "airflow", "spark",
    "terraform", "helm", "mlflow", "kubeflow", "celery", "redis",
    "monitoring", "observability", "feature store", "feature engineering",
    "data pipeline", "etl",
]

RESEARCH_ONLY_SIGNALS = [
    "pure research", "research lab", "theoretical", "academic", "university lab",
    "published paper", "conference paper", "workshop paper",
]

STRATEGY_ONLY_SIGNALS = [
    "strategized", "stakeholder management", "roadmap planning",
    "designed the vision", "presented to", "coordinated with",
]

SHALLOW_AI_SIGNALS = [
    "langchain", "openai api", "chatgpt wrapper", "gpt wrapper",
    "llm api", "prompt engineering",
    "chatgpt", "gpt-4", "gpt4", "claude api", "anthropic api",
    "gemini api", "cohere api", "ai wrapper", "api wrapper",
]

TOP_N = 100
REASONING_MAX_LEN = 300
