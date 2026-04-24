import csv
import os
import random
import psycopg2
from dotenv import load_dotenv

# Load environment variables
# We load from the current directory explicitly to be safe
dotenv_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path=dotenv_path)

DB_HOST = os.getenv("BEC_SQL_HOST")
DB_USER = os.getenv("BEC_SQL_USER")
DB_PASSWORD = os.getenv("BEC_SQL_PASSWORD")
DB_NAME = os.getenv("BEC_SQL_DATABASE")
DB_PORT = os.getenv("BEC_SQL_PORT", "5432")

INPUT_FILE = "requests/batch1.csv"
OUTPUT_FILE = "requests/batch1_filled.csv"
UNMATCHED_FILE = "requests/batch1_unmatched.csv"

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
        port=DB_PORT
    )

def main():
    print(f"Reading {INPUT_FILE}...")
    rows = []
    try:
        with open(INPUT_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                # Assuming format: bdrc_w_id, bdrc_i_id
                if len(row) >= 2:
                    rows.append((row[0].strip(), row[1].strip()))
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Found {len(rows)} rows.")

    # Dictionary to store results: (w_id, i_id) -> list of {s3_etag, job_name}
    results_map = {}
    
    chunk_size = 500
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i:i+chunk_size]
            
            # Create a values string like (%s, %s), (%s, %s), ...
            placeholders = ",".join(["(%s, %s)"] * len(chunk))
            
            query = f"""
            WITH input_pairs (w_id, i_id) AS (
                VALUES {placeholders}
            )
            SELECT 
                ip.w_id, 
                ip.i_id, 
                te.s3_etag, 
                j.name
            FROM input_pairs ip
            JOIN volumes v ON v.bdrc_w_id = ip.w_id AND v.bdrc_i_id = ip.i_id
            JOIN task_executions te ON te.volume_id = v.id
            JOIN jobs j ON j.id = te.job_id
            WHERE te.job_id IN (4, 5)
            """
            
            # Flatten chunk for params
            params = [item for sublist in chunk for item in sublist]
            
            cur.execute(query, params)
            
            fetched = cur.fetchall()
            for w_id, i_id, s3_etag, job_name in fetched:
                key = (w_id, i_id)
                if key not in results_map:
                    results_map[key] = []
                
                # Handle bytea for s3_etag
                etag_str = ""
                if isinstance(s3_etag, (bytes, memoryview)):
                    etag_str = bytes(s3_etag).hex()
                else:
                    etag_str = str(s3_etag)
                
                # Truncate to first 6 chars
                etag_str = etag_str[:6]
                
                results_map[key].append({
                    "s3_etag": etag_str,
                    "name": job_name
                })
                
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")
        return

    print("Processing results and writing output...")
    
    match_count = 0
    with open(OUTPUT_FILE, "w", newline="") as f_out, open(UNMATCHED_FILE, "w", newline="") as f_unmatched:
        writer_out = csv.writer(f_out)
        writer_unmatched = csv.writer(f_unmatched)
        
        for w_id, i_id in rows:
            key = (w_id, i_id)
            candidates = results_map.get(key, [])
            
            if not candidates:
                # No match found for job 4 or 5
                writer_unmatched.writerow([w_id, i_id])
            else:
                # Pick one randomly
                choice = random.choice(candidates)
                writer_out.writerow([w_id, i_id, choice["s3_etag"], choice["name"]])
                match_count += 1
                
    print(f"Done. Matches written to {OUTPUT_FILE}, unmatched to {UNMATCHED_FILE}.")
    print(f"Matches found for {match_count}/{len(rows)} rows.")

if __name__ == "__main__":
    main()
