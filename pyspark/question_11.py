from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    udf, col, regexp_replace, lower, split, size, explode, collect_list, 
    count, lit, log, desc, asc, array, when, row_number
)
from pyspark.sql.types import ArrayType, StringType, FloatType, MapType, DoubleType, StructType, StructField
from pyspark.sql.window import Window
import re

print("Initializing Spark Session (PRACTICAL VERSION)...")
spark = SparkSession.builder \
    .appName("Q11: TF-IDF PRACTICAL") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.driver.maxResultSize", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("Spark Session created!\n")


print("="*80)
print("STEP 1: Loading Books")
print("="*80)

from pyspark.sql.functions import regexp_extract

books_df = spark.read.text("dataset/*.txt", wholetext=True) \
    .selectExpr("input_file_name() as file_path", "value as text")

books_df = books_df.withColumn(
    "file_name", 
    regexp_extract(col("file_path"), r"([^/\\]+\.txt)$", 1)
)

total_books = books_df.count()
print(f"Total books loaded: {total_books}\n")


print("="*80)
print("STEP 2: Preprocessing - Removing Project Gutenberg Headers/Footers")
print("="*80)

def remove_gutenberg_header_footer(text):
    if not text:
        return ""
    
    start_markers = [
        r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK",
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK"
    ]
    
    start_pos = 0
    for marker in start_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            start_pos = text.find('\n', match.end()) + 1
            break
    
    end_markers = [
        r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK",
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK"
    ]
    
    end_pos = len(text)
    for marker in end_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            end_pos = match.start()
            break
    
    clean_text = text[start_pos:end_pos]
    return clean_text.strip()

remove_header_footer_udf = udf(remove_gutenberg_header_footer, StringType())
books_cleaned = books_df.withColumn("clean_text", remove_header_footer_udf(col("text")))

print("Headers and footers removed.\n")


print("="*80)
print("STEP 3: Text Preprocessing")
print("="*80)

books_cleaned = books_cleaned.withColumn("clean_text", lower(col("clean_text")))
print("Converted to lowercase")

books_cleaned = books_cleaned.withColumn(
    "clean_text", 
    regexp_replace(col("clean_text"), r"[^a-z\s]", " ")
)
print("Removed punctuation and numbers")

books_cleaned = books_cleaned.withColumn(
    "clean_text", 
    regexp_replace(col("clean_text"), r"\s+", " ")
)
print("Removed extra whitespace")

books_cleaned = books_cleaned.withColumn(
    "words_raw",
    split(col("clean_text"), " ")
)

def filter_words(words):
    if not words:
        return []
    return [w for w in words if len(w) >= 3]

filter_words_udf = udf(filter_words, ArrayType(StringType()))
books_cleaned = books_cleaned.withColumn("words_filtered", filter_words_udf(col("words_raw")))

print("Tokenized into words (minimum length: 3 characters)")

stopwords = set([
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 
    'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new',
    'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put',
    'say', 'she', 'too', 'use', 'had', 'may', 'any', 'from', 'have', 'that',
    'this', 'with', 'they', 'will', 'your', 'about', 'could', 'there', 'their',
    'which', 'would', 'these', 'other', 'than', 'then', 'them', 'been', 'into',
    'such', 'only', 'some', 'time', 'very', 'what', 'when', 'where', 'were'
])

def remove_stopwords(words):
    if not words:
        return []
    return [w for w in words if w not in stopwords]

remove_stopwords_udf = udf(remove_stopwords, ArrayType(StringType()))
books_cleaned = books_cleaned.withColumn("words", remove_stopwords_udf(col("words_filtered")))

print("Removed stop words")

# ADD WORD COUNT COLUMN
books_cleaned = books_cleaned.withColumn("word_count", size(col("words")))

stats = books_cleaned.agg(
    count("file_name").alias("total_books")
).first()

print(f"\nSTATISTICS:")
print(f"  Total books: {stats['total_books']}")

print("\nSample of preprocessed data:")
books_cleaned.select("file_name", "words").show(3, truncate=80)

books_processed = books_cleaned.select("file_name", "words", "word_count")


print("\n" + "="*80)
print("STEP 4: MANUAL TF-IDF CALCULATION")
print("="*80)

print("\n4.1: Creating word-document pairs...")
words_exploded = books_processed.select(
    "file_name",
    "word_count",
    explode("words").alias("word")
)

print("4.2: Calculating Term Frequency (TF)...")
word_counts = words_exploded.groupBy("file_name", "word").agg(
    count("word").alias("term_count")
)

tf_df = word_counts.join(
    books_processed.select("file_name", "word_count"),
    "file_name"
).withColumn(
    "tf",
    col("term_count") / col("word_count")
)

print(f"TF calculated")

print("\n4.3: Calculating Inverse Document Frequency (IDF)...")
document_frequency = word_counts.groupBy("word").agg(
    count("file_name").alias("doc_freq")
)

total_documents = books_processed.count()

idf_df = document_frequency.withColumn(
    "idf",
    log(lit(total_documents) / col("doc_freq"))
)

print(f"IDF calculated")

print("\n4.4: Computing TF-IDF scores...")
tfidf_df = tf_df.join(idf_df, "word").withColumn(
    "tfidf",
    col("tf") * col("idf")
)

print(f"TF-IDF calculated")

print("\n" + "="*80)
print("VALIDATION: Top 20 TF-IDF words for '10.txt'")
print("="*80)

top_tfidf_words = tfidf_df.filter(col("file_name") == "10.txt") \
    .orderBy(desc("tfidf")) \
    .select("word", "tf", "idf", "tfidf") \
    .limit(20)

top_tfidf_words.show(20, truncate=False)

# Cache TF-IDF
tfidf_df.cache()
tfidf_df.count()


print("\n" + "="*80)
print("STEP 5: SMART ALL-PAIRS SIMILARITY (Memory Efficient)")
print("="*80)

# Create sparse vectors
vocabulary = idf_df.select("word").rdd.map(lambda row: row['word']).collect()
vocab_size = len(vocabulary)
word_to_index = {word: idx for idx, word in enumerate(vocabulary)}

print(f"Vocabulary size: {vocab_size:,} unique words")

word_to_index_bc = spark.sparkContext.broadcast(word_to_index)

def create_tfidf_vector(words_tfidf):
    word_idx_map = word_to_index_bc.value
    vector = {}
    for word, tfidf in words_tfidf:
        if word in word_idx_map:
            idx = word_idx_map[word]
            vector[idx] = float(tfidf)
    return vector

book_vectors = tfidf_df.groupBy("file_name").agg(
    collect_list(array(col("word"), col("tfidf"))).alias("word_tfidf_list")
)

create_vector_udf = udf(create_tfidf_vector, MapType(StringType(), DoubleType()))

book_vectors = book_vectors.withColumn(
    "tfidf_vector",
    create_vector_udf(col("word_tfidf_list"))
)

print("Vector representation created")

book_vectors.cache()
book_vectors.count()


def cosine_similarity(vec1_dict, vec2_dict):
    if not vec1_dict or not vec2_dict:
        return 0.0
    
    dot_product = 0.0
    for idx, val1 in vec1_dict.items():
        if idx in vec2_dict:
            dot_product += val1 * vec2_dict[idx]
    
    magnitude1 = sum(v * v for v in vec1_dict.values()) ** 0.5
    magnitude2 = sum(v * v for v in vec2_dict.values()) ** 0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

print("\n" + "="*80)
print("Computing Similarities in BATCHES (won't freeze)")
print("="*80)

# Get all book names
all_books = book_vectors.select("file_name").rdd.map(lambda r: r[0]).collect()

book_vectors_collected = book_vectors.collect()
book_vector_map = {row['file_name']: row['tfidf_vector'] for row in book_vectors_collected}

print(f"\nTotal books: {len(all_books)}")
print("Computing all pairwise similarities...")


from itertools import combinations

similarities = []
batch_size = 1000
count = 0

# Use combinations to get unique pairs
for book1, book2 in combinations(all_books, 2):
    vec1 = book_vector_map.get(book1)
    vec2 = book_vector_map.get(book2)
    
    if vec1 and vec2:
        sim = cosine_similarity(vec1, vec2)
        similarities.append((book1, book2, float(sim)))
        count += 1
        
        if count % batch_size == 0:
            print(f"Processed {count:,} pairs...")

print(f"\nComputed {len(similarities):,} pairwise similarities")

# Convert to DataFrame
schema = StructType([
    StructField("book1", StringType(), False),
    StructField("book2", StringType(), False),
    StructField("similarity", FloatType(), False)
])

all_similarities = spark.createDataFrame(similarities, schema)


print("\n" + "="*80)
print("STEP 6: Top 5 Similar Books for '10.txt'")
print("="*80)

target_book = "10.txt"

target_similarities = all_similarities.filter(
    (col("book1") == target_book) | (col("book2") == target_book)
).withColumn(
    "other_book",
    when(col("book1") == target_book, col("book2")).otherwise(col("book1"))
)

top_5 = target_similarities.select("other_book", "similarity") \
    .orderBy(desc("similarity")) \
    .limit(5)

print(f"\nTop 5 books most similar to '{target_book}':\n")
results = top_5.collect()

for i, row in enumerate(results, 1):
    print(f"  {i}. {row['other_book']:<20} Similarity: {row['similarity']:.6f}")


print("\n" + "="*80)
print("STEP 7: Top 20 Most Similar Book Pairs")
print("="*80)

top_pairs = all_similarities.orderBy(desc("similarity")).limit(20)
print("\n")
top_pairs.show(20, truncate=False)


print("\n" + "="*80)
print("STEP 8: Saving Results")
print("="*80)

# Save TF-IDF scores
tfidf_df.select("file_name", "word", "tf", "idf", "tfidf") \
    .write.mode("overwrite").parquet("output/tfidf_scores.parquet")
print("TF-IDF scores saved to: output/tfidf_scores.parquet")

# Save all pairwise similarities
all_similarities.write.mode("overwrite").parquet("output/all_pairwise_similarities.parquet")
print("All pairwise similarities saved to: output/all_pairwise_similarities.parquet")

# Save top 5 for target
top_5.coalesce(1).write.mode("overwrite").csv(
    "output/top_5_similar_to_10.csv",
    header=True
)
print("Top 5 similar books to 10.txt saved to: output/top_5_similar_to_10.csv")

# Save top 20 pairs
top_pairs.coalesce(1).write.mode("overwrite").csv(
    "output/top_20_similar_pairs.csv",
    header=True
)
print("Top 20 similar pairs saved to: output/top_20_similar_pairs.csv")

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\nTotal book pairs computed: {len(similarities):,}")
print(f"Average similarity: {sum(s[2] for s in similarities) / len(similarities):.6f}")

print("\n" + "="*80)
print("Question 11 Complete")
print("="*80)

spark.stop()