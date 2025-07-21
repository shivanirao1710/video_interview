import os
import logging
import psycopg2
from transformers import pipeline
from tqdm import tqdm

# === Setup Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Load username from environment ===
USERNAME = os.getenv("LOGGED_IN_USER", "unknown")

if not USERNAME or USERNAME == "unknown":
    logging.error("LOGGED_IN_USER not found. Aborting report generation.")
    exit()

# === Load LLM pipeline ===
try:
    pipe = pipeline("text2text-generation", model="google/flan-t5-base")
    logging.info("LLM model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load LLM model: {e}")
    exit()

# === Database config ===
DB_NAME = "behavioural"
DB_USER = "postgres"
DB_PASSWORD = "shivanirao1710"
DB_HOST = "localhost"
DB_PORT = "5432"

# === Connect to DB and process ===
try:
    with psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    ) as conn:
        with conn.cursor() as cur:

            # Fetch user's interview answers
            cur.execute("SELECT question, answer FROM interview_answers WHERE username = %s", (USERNAME,))
            rows = cur.fetchall()

            if not rows:
                logging.warning("No interview answers found.")
                exit()

            # Delete previous report
            cur.execute("DELETE FROM soft_skill_analysis WHERE username = %s", (USERNAME,))
            logging.info("Old soft skill analysis deleted.")

            # Analyze and store in database only
            for question, answer in tqdm(rows, desc="Analyzing answers"):
                if not answer.strip():
                    continue

                if len(answer.split()) > 200:
                    answer = " ".join(answer.split()[:200]) + "..."

                prompt = (
                    f"You are a behavioral analyst. Given the interview answer below, "
                    f"identify observed soft skills, strengths, and areas for improvement:\n\n"
                    f"Answer: \"{answer}\"\n\n"
                    f"Respond in a structured format."
                )

                try:
                    result = pipe(prompt, max_new_tokens=150)[0]["generated_text"]
                except Exception as e:
                    logging.error(f"Error during LLM generation: {e}")
                    result = "Analysis unavailable due to processing error."

                cur.execute("""
                    INSERT INTO soft_skill_analysis (username, question, answer, analysis)
                    VALUES (%s, %s, %s, %s)
                """, (USERNAME, question.strip(), answer.strip(), result.strip()))

            conn.commit()
            logging.info("✅ Report generation complete. Data stored in database.")

except Exception as e:
    logging.error(f"Database or processing error: {e}")